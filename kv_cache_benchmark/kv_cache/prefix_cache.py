"""
Hierarchical prefix caching for KV Cache Benchmark.

Models the reuse of common prompts (e.g., system prompts) across
users to reduce redundant cache allocations.
"""

import hashlib
import random
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Set, Tuple
from datetime import datetime
from enum import Enum

from kv_cache.config import cfg
from kv_cache.models import ModelConfig, InferenceRequest


class PrefixType(Enum):
    """Enumeration for the different tiers of prefix caching."""
    SYSTEM_PROMPT = "system_prompt"
    COMMON_PHRASE = "common_phrase"
    USER_SPECIFIC = "user_specific"


@dataclass
class PrefixCacheEntry:
    """Represents a cached prefix."""
    prefix_key: str
    prefix_type: PrefixType
    text_hash: str
    token_count: int
    kv_cache_key: str

    # Usage statistics to track popularity and reuse.
    use_count: int = 0
    first_seen: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    users_using: Set[str] = field(default_factory=set)

    # Storage information.
    storage_tier: str = ""
    size_bytes: int = 0


class PrefixMatcher:
    """Detects and matches common prefixes in requests to enable reuse."""

    COMMON_SYSTEM_PROMPTS = [
        "You are a helpful assistant.",
        "You are an AI assistant helping with coding tasks.",
        "You are a professional writing assistant.",
    ]

    def __init__(self, min_prefix_length: int = None):
        self.min_prefix_length = min_prefix_length if min_prefix_length is not None else cfg('prefix_cache', 'min_prefix_length', default=50)
        self.prefix_index: Dict[str, PrefixCacheEntry] = {}
        self.prefix_frequency: Dict[str, int] = {}
        self.lock = threading.Lock()

    def hash_prefix(self, text: str, token_count: int) -> str:
        """Creates a deterministic hash for a given text prefix."""
        content = f"{text[:500]}_{token_count}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def detect_system_prompt(self, context_tokens: int) -> Optional[PrefixCacheEntry]:
        """Simulates the detection of a common system prompt at the start of a request."""
        system_prompt_hit_probability = cfg('prefix_cache', 'system_prompt_hit_probability', default=0.2)
        if random.random() < system_prompt_hit_probability:
            system_prompt = random.choice(self.COMMON_SYSTEM_PROMPTS)
            prefix_hash = self.hash_prefix(system_prompt, len(system_prompt.split()))

            with self.lock:
                if prefix_hash in self.prefix_index:
                    entry = self.prefix_index[prefix_hash]
                    entry.use_count += 1
                    entry.last_used = datetime.now()
                    return entry
                else:
                    entry = PrefixCacheEntry(
                        prefix_key=f"system_{prefix_hash}",
                        prefix_type=PrefixType.SYSTEM_PROMPT,
                        text_hash=prefix_hash,
                        token_count=len(system_prompt.split()),
                        kv_cache_key=f"kv_system_{prefix_hash}",
                        use_count=1
                    )
                    self.prefix_index[prefix_hash] = entry
                    return entry
        return None


class PrefixCacheManager:
    """Orchestrates the prefix matching and caching logic."""

    def __init__(self, cache, max_prefix_entries: int = None):
        self.cache = cache
        self.max_prefix_entries = max_prefix_entries if max_prefix_entries is not None else cfg('prefix_cache', 'max_prefix_entries', default=1000)
        self.prefix_matcher = PrefixMatcher()
        self.lock = threading.Lock()

        self.stats = {
            'prefix_hits': 0,
            'prefix_misses': 0,
            'system_prompt_reuse': 0,
            'common_phrase_reuse': 0,
            'bytes_saved': 0
        }

    def record_prefix_result(
        self,
        prefix_entry: Optional[PrefixCacheEntry],
        model_config: ModelConfig
    ):
        """Record prefix-cache metrics for a completed request."""
        with self.lock:
            if prefix_entry:
                self.stats['prefix_hits'] += 1
                if prefix_entry.prefix_type == PrefixType.SYSTEM_PROMPT:
                    self.stats['system_prompt_reuse'] += 1
                self.stats['bytes_saved'] += prefix_entry.token_count * model_config.kv_cache_size_per_token
            else:
                self.stats['prefix_misses'] += 1

    def check_prefix_cache(
        self,
        request: InferenceRequest,
        model_config: ModelConfig,
        record_stats: bool = True
    ) -> Tuple[Optional[PrefixCacheEntry], int]:
        """
        Checks if the beginning of a request matches a known, cached prefix.

        Returns:
            A tuple containing the PrefixCacheEntry if a hit occurs (or None),
            and the number of remaining (non-prefixed) tokens in the request.
        """
        prefix_entry = self.prefix_matcher.detect_system_prompt(request.context_tokens)

        if prefix_entry:
            if record_stats:
                self.record_prefix_result(prefix_entry, model_config)

            remaining_tokens = max(0, request.context_tokens - prefix_entry.token_count)
            return prefix_entry, remaining_tokens
        else:
            if record_stats:
                self.record_prefix_result(None, model_config)
            return None, request.context_tokens
