#!/usr/bin/env python3
"""
KV Cache Benchmark - Multi-Tier Performance Comparison
Hazem Awadallah, Kingston Digital, 2025
Assisted by Github Copilot

Integrated Multi-User KV Cache Benchmark - Enhanced Version
MLPerf Storage Working Group - Benchmark Implementation

This script provides a comprehensive, configurable benchmark for testing storage system
performance for Large Language Model (LLM) Key-Value (KV) cache offloading. It simulates
a realistic multi-tenant inference environment with a sophisticated multi-tier cache.

--- Key Features ---
1.  Phase-Aware Processing: Differentiates between the write-heavy 'prefill' phase
    and the read-heavy 'decode' phase.
2.  Stateful Multi-turn Conversations: Models cache reuse in conversational AI.
3.  Hierarchical Prefix Caching: Simulates the caching of common prompts (e.g., system prompts)
    for high-efficiency reuse across users.
4.  RAG Workload Modeling: Simulates Retrieval-Augmented Generation workloads, which involve
    large context sizes and unique I/O patterns.
5.  Adaptive Autoscaling: Automatically adjusts the user load to find the saturation point
    of the storage system.
6.  Trace-Driven Validation: Can validate its own simulation against real-world traces.
7.  QoS Support: Implements different priority levels (Interactive, Responsive, Batch) to
    mimic real-world request scheduling.
8.  Enhanced Metrics and Reporting: Provides detailed statistics on latency, throughput, IOPS,
    and cache performance across all tiers.

Target Accuracy: Â±5% representation of real LLM inference clusters
"""

import os
import sys
import time
import json
import tempfile
import numpy as np
import hashlib
import shutil
from pathlib import Path
from dataclasses import dataclass, asdict, field, is_dataclass
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import threading
import queue
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import argparse
import csv

# Attempt to import optional GPU libraries (torch, cupy)
# The benchmark can run in a CPU-only environment if these are not found.
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


# ============================================================================ 
# CORE DATA MODELS
# Defines the basic data structures used throughout the benchmark.
# ============================================================================ 

@dataclass
class ModelConfig:
    """
    Configuration for a model's KV cache requirements.

    This dataclass holds the architectural parameters of an LLM that are essential
    for calculating the size of its KV cache.
    """
    name: str
    num_layers: int         # Number of transformer layers in the model.
    hidden_dim: int         # The size of the main hidden state vector.
    num_heads: int          # Number of attention heads for queries (Q).
    kv_heads: int           # Number of attention heads for keys/values (K/V). For GQA, kv_heads < num_heads.
    dtype: str = 'float16'  # Data type used for cache tensors (e.g., float16, bfloat16).

    @property
    def bytes_per_element(self) -> int:
        """Returns the size in bytes of a single element based on the data type."""
        dtype_map = {'float32': 4, 'float16': 2, 'bfloat16': 2, 'int8': 1}
        return dtype_map.get(self.dtype, 2) # Default to 2 bytes for float16/bfloat16

    @property
    def kv_dim_per_head(self) -> int:
        """Calculates the dimension of each Key/Value attention head."""
        return self.hidden_dim // self.num_heads

    @property
    def kv_cache_size_per_token(self) -> int:
        """
        Calculates the total memory in bytes required to store the KV cache for a single token.
        This is the fundamental unit for all memory calculations in the benchmark.
        Formula: num_layers * num_kv_heads * head_dimension * 2 (for K and V) * bytes_per_element
        """
        return self.num_layers * self.kv_heads * self.kv_dim_per_head * 2 * self.bytes_per_element


# A dictionary of pre-defined model configurations that can be selected via command line.
MODEL_CONFIGS = {
    'tiny-1b': ModelConfig(
        name='Tiny 1B',
        num_layers=12,
        hidden_dim=1024,
        num_heads=8,
        kv_heads=4,
        dtype='float16'
    ),
    'mistral-7b': ModelConfig(
        name='Mistral 7B',
        num_layers=32,
        hidden_dim=4096,
        num_heads=32,
        kv_heads=8,
        dtype='float16'
    ),
    'llama2-7b': ModelConfig(
        name='Llama 2 7B',
        num_layers=32,
        hidden_dim=4096,
        num_heads=32,
        kv_heads=32, # Llama 2 uses Multi-Head Attention (MHA), so kv_heads == num_heads
        dtype='float16'
    ),
    'llama3.1-8b': ModelConfig(
        name='Llama 3.1 8B',
        num_layers=32,
        hidden_dim=4096,
        num_heads=32,
        kv_heads=8,
        dtype='float16'
    ),
    'llama3.1-70b-instruct': ModelConfig(
        name='Llama 3.1 70B Instruct',
        num_layers=80,
        hidden_dim=8192,
        num_heads=64,
        kv_heads=8,
        dtype='float16'
    ),
}


# ============================================================================ 
# FEATURE 1: PHASE-AWARE PROCESSING
# Models the two distinct phases of LLM inference, which have different I/O patterns.
# ============================================================================ 

class InferencePhase(Enum):
    """Enumeration for the two main phases of LLM inference."""
    PREFILL = "prefill"      # Write-heavy phase: processing the input prompt.
    DECODE = "decode"        # Read-heavy phase: generating output tokens one by one.
    PREFILL_DECODE = "both"  # A combined phase for very short requests.


class GenerationMode(Enum):
    """Enumeration for token generation simulation modes."""
    NONE = "none"           # Pure storage benchmark. No simulated sleep. Latency is 100% I/O.
    FAST = "fast"           # Simulates a very fast GPU (2ms/token) to model some backpressure.
    REALISTIC = "realistic" # Simulates a realistic GPU (30ms/token) for end-to-end latency analysis.

# Defines the sleep time per token to simulate GPU work for each mode.
GENERATION_TIMING = {
    GenerationMode.NONE: 0.0,
    GenerationMode.FAST: 0.002,
    GenerationMode.REALISTIC: 0.030,
}


# ============================================================================ 
# FEATURE 7: QOS SUPPORT
# Models a multi-tenant environment where requests have different priorities.
# ============================================================================ 

class QoSLevel(Enum):
    """Enumeration for Quality of Service (QoS) levels, defining user priority."""
    INTERACTIVE = "interactive" # Highest priority, for real-time applications (e.g., chatbot UI).
    RESPONSIVE = "responsive"   # High priority, for near real-time tasks.
    BATCH = "batch"             # Low priority, for offline processing.


@dataclass
class QoSSLA:
    """
    Represents a Service Level Agreement (SLA) for a given QoS level.
    Defines the performance targets and tracks violations.
    """
    qos_level: QoSLevel
    target_latency_p95_ms: float # The 95th percentile latency target.
    target_latency_p99_ms: float # The 99th percentile latency target.
    priority: int                  # An integer priority level (higher is more important).

    # SLA violation tracking
    violations: int = 0
    total_requests: int = 0

    @property
    def sla_compliance(self) -> float:
        """Calculates the percentage of requests that met the SLA target."""
        if self.total_requests == 0:
            return 1.0
        return 1.0 - (self.violations / self.total_requests)


# Pre-defined QoS profiles mapping each level to a specific SLA.
QOS_PROFILES = {
    QoSLevel.INTERACTIVE: QoSSLA(
        qos_level=QoSLevel.INTERACTIVE,
        target_latency_p95_ms=50,
        target_latency_p99_ms=100,
        priority=3
    ),
    QoSLevel.RESPONSIVE: QoSSLA(
        qos_level=QoSLevel.RESPONSIVE,
        target_latency_p95_ms=100,
        target_latency_p99_ms=200,
        priority=2
    ),
    QoSLevel.BATCH: QoSSLA(
        qos_level=QoSLevel.BATCH,
        target_latency_p95_ms=1000,
        target_latency_p99_ms=5000,
        priority=1
    )
}


@dataclass
class UserProfile:
    """Represents a simulated user with specific behavior patterns."""
    user_id: str
    context_length: int      # The number of tokens in the user's prompts.
    generation_length: int   # The number of tokens the user requests to be generated.
    think_time: float        # The simulated time the user "thinks" between requests.
    priority: int
    qos_level: QoSLevel
    session_start: datetime = field(default_factory=datetime.now)
    total_latency: float = 0.0
    request_count: int = 0


@dataclass
class InferenceRequest:
    """Represents a single, atomic inference request sent to the benchmark."""
    user_id: str
    request_id: str
    timestamp: datetime
    context_tokens: int
    generate_tokens: int
    priority: int
    phase: InferencePhase = InferencePhase.PREFILL_DECODE
    qos_level: QoSLevel = QoSLevel.BATCH
    cache_key: Optional[str] = None # The unique identifier for this request's KV cache.

    # Timing fields to track latency at different stages.
    submit_time: float = field(default_factory=time.perf_counter) # When the request was created.
    start_time: float = 0      # When processing began.
    complete_time: float = 0   # When processing finished.

    # Conversation tracking for stateful workloads.
    conversation_id: Optional[str] = None
    turn_number: int = 0

    def __post_init__(self):
        """Post-initialization hook to automatically generate a cache key.

        If a `cache_key` is not explicitly provided during the object's
        creation, this method constructs one based on the available context.

        The generation logic is as follows:
        - If a `conversation_id` is present, the key is formatted as
            `f"{conversation_id}_turn_{turn_number}"` to uniquely identify a
            specific turn within a conversation.
        - Otherwise, it defaults to a user-specific context key formatted as
            `f"{user_id}_ctx"`.

        This ensures that every instance has a non-null `cache_key` for
        cache management.
        """
       
        if self.cache_key is None:
            if self.conversation_id:
                self.cache_key = f"{self.conversation_id}_turn_{self.turn_number}"
            else:
                self.cache_key = f"{self.user_id}_ctx"

    @property
    def total_latency_ms(self) -> float:
        """Calculates the total end-to-end latency for the request in milliseconds."""
        if self.complete_time == 0:
            return 0
        return (self.complete_time - self.submit_time) * 1000


# ============================================================================ 
# FEATURE 2: STATEFUL MULTI-TURN CONVERSATIONS
# Models how conversational context is managed and reused over time.
# ============================================================================ 

@dataclass
class ConversationState:
    """Tracks the state of a single multi-turn conversation for a user."""
    conversation_id: str
    user_id: str
    turn_number: int
    created_at: datetime
    last_access: datetime

    # KV cache management for this conversation.
    cache_keys: List[str] = field(default_factory=list) # List of cache keys for each turn.
    cumulative_tokens: int = 0
    cache_locations: Dict[str, str] = field(default_factory=dict)

    # Metadata for advanced caching strategies.
    system_prompt_key: Optional[str] = None
    common_prefix_keys: List[str] = field(default_factory=list)

    # Performance tracking for this conversation.
    turns_completed: int = 0
    total_latency: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0


class ConversationManager:
    """Manages the lifecycle of all multi-turn conversations and enables cache reuse."""

    def __init__(self, max_conversations: int = 1000, max_turns_per_conv: int = 50):
        self.conversations: Dict[str, ConversationState] = {}
        self.max_conversations = max_conversations
        self.max_turns_per_conv = max_turns_per_conv
        self.lock = threading.Lock() # Protects access to the shared conversations dictionary.

    def start_conversation(self, user_id: str, system_prompt: Optional[str] = None) -> str:
        """Initializes a new conversation for a given user.
        This method creates a unique conversation ID and a corresponding
        `ConversationState` object to track the conversation's progress.
        It handles an optional system prompt by creating a reusable, hashed key for it.
        If the total number of active conversations reaches the configured
        maximum (`self.max_conversations`), the least recently accessed
        conversation is evicted to make room for the new one.
        Args:
            user_id (str): The unique identifier for the user starting the conversation.
            system_prompt (Optional[str]): An optional initial prompt to set the
                conversation's context. Defaults to None.
        Returns:
            str: The unique identifier generated for the new conversation.
        """
       
        conv_id = f"conv_{user_id}_{int(time.time()*1000)}"

        state = ConversationState(
            conversation_id=conv_id,
            user_id=user_id,
            turn_number=0,
            created_at=datetime.now(),
            last_access=datetime.now(),
            cache_keys=[],
            cumulative_tokens=0,
            cache_locations={}
        )

        # If a system prompt is provided, create a deterministic, reusable key for it.
        # Hashing the prompt text ensures that identical system prompts across different
        # conversations map to the same cache key, enabling high-efficiency reuse.
        if system_prompt:
            state.system_prompt_key = f"system_prompt_{hashlib.sha256(system_prompt.encode()).hexdigest()[:16]}"

        with self.lock:
            # If the number of conversations exceeds the max, evict the oldest one. Otherwise, add the new conversation.
            if len(self.conversations) >= self.max_conversations:
                self._evict_oldest_conversation()

            self.conversations[conv_id] = state

        return conv_id

    def add_turn(self, conversation_id: str, user_message_tokens: int,
                 assistant_response_tokens: int) -> Tuple[int, str]:
        """
        Adds a new turn to an existing conversation, updating its state.
        This method is thread-safe. It locates a conversation by its ID,
        increments the turn counter, updates the total token count, and generates
        a unique cache key for the new turn. The conversation's last access
        time is also updated.
        Args:
            conversation_id (str): The unique identifier for the conversation.
            user_message_tokens (int): The number of tokens in the user's message for this turn.
            assistant_response_tokens (int): The number of tokens in the assistant's response for this turn.
         Returns:
            Tuple[int, str]: A tuple containing the new turn number and the unique cache key generated for this turn.
        Raises:
            ValueError: If no conversation with the given `conversation_id` is found.
        """
        
        with self.lock:
            if conversation_id not in self.conversations:
                raise ValueError(f"Conversation {conversation_id} not found")

            state = self.conversations[conversation_id]
            state.turn_number += 1
            state.last_access = datetime.now()

            turn_cache_key = f"{conversation_id}_turn_{state.turn_number}"

            # Update conversation state with new tokens and cache key.
            state.cache_keys.append(turn_cache_key)
            state.cumulative_tokens += user_message_tokens + assistant_response_tokens
            state.turns_completed += 1

            return state.turn_number, turn_cache_key

    def get_conversation_context_size(self, conversation_id: str) -> int:
        """Gets the total number of tokens accumulated in a conversation."""
        with self.lock:
            if conversation_id not in self.conversations:
                return 0
            return self.conversations[conversation_id].cumulative_tokens

    def get_all_previous_turn_keys(self, conversation_id: str, current_turn: int) -> List[str]:
        """
        Retrieves all cache keys from previous turns in a conversation.

        This method is used to assemble the full context for a new turn by fetching
        the cache keys for all preceding turns in a given conversation. It allows
        the inference engine to load the entire conversational history from the
        KV cache before processing the new user input.

        Args:
            conversation_id (str): The unique identifier for the conversation.
            current_turn (int): The current turn number. The cache key for this
                    turn will be excluded from the result.

        Returns:
            List[str]: A list of cache keys corresponding to all turns before
                   the current one. Returns an empty list if the conversation
                   is not found.
        """
        with self.lock:
            if conversation_id not in self.conversations:
                return []
            state = self.conversations[conversation_id]
            # Return all turns up to (but not including) the current turn
            return [key for key in state.cache_keys if key != f"{conversation_id}_turn_{current_turn}"]

    def _evict_oldest_conversation(self):
        """Evicts the least recently used (LRU) conversation to make space."""
        if not self.conversations:
            return
        # Find the conversation with the oldest `last_access` timestamp (Least Recently Used).
        # The min() function scans all conversations to find the one with the smallest
        # (oldest) `last_access` time. This is the LRU entry.
        #
        #    Time -->
        #    +------------------------------------------------+
        #    | Conv_B |  Conv_D  |   Conv_A   |     Conv_C     |
        #    +------------------------------------------------+
        #    ^
        #    |
        #  Oldest Access Time (min). This one is evicted.
        #
        oldest_conv_id = min(
            self.conversations,
            key=lambda k: (self.conversations[k].last_access, self.conversations[k].created_at)
        )
        del self.conversations[oldest_conv_id]


# ============================================================================ 
# FEATURE 3: HIERARCHICAL PREFIX CACHING
# Models the reuse of common prompts (e.g., "You are a helpful assistant").
# ============================================================================ 

class PrefixType(Enum):
    """Enumeration for the different tiers of prefix caching."""
    SYSTEM_PROMPT = "system_prompt"      # Highest reuse, almost never evicted.
    COMMON_PHRASE = "common_phrase"      # High reuse, rarely evicted.
    USER_SPECIFIC = "user_specific"      # Low reuse, normal eviction policy.


@dataclass
class PrefixCacheEntry:
    """Represents a cached prefix."""
    prefix_key: str
    prefix_type: PrefixType
    text_hash: str
    token_count: int
    kv_cache_key: str # The key pointing to the actual KV cache data in the multi-tier cache.

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

    # A list of common system prompts to simulate prefix matching.
    COMMON_SYSTEM_PROMPTS = [
        "You are a helpful assistant.",
        "You are an AI assistant helping with coding tasks.",
        "You are a professional writing assistant.",
    ]

    def __init__(self, min_prefix_length: int = 50):
        self.min_prefix_length = min_prefix_length
        self.prefix_index: Dict[str, PrefixCacheEntry] = {}
        self.prefix_frequency: Dict[str, int] = {}
        self.lock = threading.Lock()

    def hash_prefix(self, text: str, token_count: int) -> str:
        """Creates a deterministic hash for a given text prefix."""
        content = f"{text[:500]}_{token_count}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def detect_system_prompt(self, context_tokens: int) -> Optional[PrefixCacheEntry]:
        """Simulates the detection of a common system prompt at the start of a request."""
        # In this simulation, 20% of requests are assumed to start with a common system prompt.
        if random.random() < 0.2:
            system_prompt = random.choice(self.COMMON_SYSTEM_PROMPTS)
            prefix_hash = self.hash_prefix(system_prompt, len(system_prompt.split()))

            with self.lock:
                if prefix_hash in self.prefix_index:
                    # If this prompt has been seen before, increment its use count.
                    entry = self.prefix_index[prefix_hash]
                    entry.use_count += 1
                    entry.last_used = datetime.now()
                    return entry
                else:
                    # If it's a new prompt, create a new entry for it.
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

    def __init__(self, cache, max_prefix_entries: int = 1000):
        self.cache = cache # A reference to the main MultiTierCache.
        self.max_prefix_entries = max_prefix_entries
        self.prefix_matcher = PrefixMatcher()
        self.lock = threading.Lock()

        # Statistics for reporting prefix cache effectiveness.
        self.stats = {
            'prefix_hits': 0,
            'prefix_misses': 0,
            'system_prompt_reuse': 0,
            'common_phrase_reuse': 0,
            'bytes_saved': 0
        }

    def check_prefix_cache(self, request: InferenceRequest, model_config: ModelConfig) -> Tuple[Optional[PrefixCacheEntry], int]:
        """
        Checks if the beginning of a request matches a known, cached prefix.

        Returns:
            A tuple containing the PrefixCacheEntry if a hit occurs (or None),
            and the number of remaining (non-prefixed) tokens in the request.
        """
        prefix_entry = self.prefix_matcher.detect_system_prompt(request.context_tokens)

        if prefix_entry:
            # On a hit, update stats and calculate how many tokens were saved.
            with self.lock:
                self.stats['prefix_hits'] += 1
                if prefix_entry.prefix_type == PrefixType.SYSTEM_PROMPT:
                    self.stats['system_prompt_reuse'] += 1
                self.stats['bytes_saved'] += prefix_entry.token_count * model_config.kv_cache_size_per_token

            # Return the prefix entry and the number of remaining tokens to process.
            remaining_tokens = max(0, request.context_tokens - prefix_entry.token_count)
            return prefix_entry, remaining_tokens
        else:
            # On a miss, update stats and return.
            with self.lock:
                self.stats['prefix_misses'] += 1
            return None, request.context_tokens


# ============================================================================ 
# FEATURE 4: RAG WORKLOAD MODELING
# Simulates a Retrieval-Augmented Generation workload, where large document
# chunks are loaded into the context window, stressing the cache.
# ============================================================================ 

@dataclass
class RAGChunk:
    """Represents a single chunk of a document in a RAG system."""
    chunk_id: str
    doc_id: str
    chunk_index: int
    token_count: int
    kv_cache_key: str # The key for this chunk's KV cache.

    access_count: int = 0
    last_accessed: datetime = field(default_factory=datetime.now)
    storage_tier: str = ""
    size_bytes: int = 0


@dataclass
class RAGDocument:
    """Represents a document that has been chunked for RAG."""
    doc_id: str
    total_tokens: int
    chunk_size: int
    chunks: List[RAGChunk] = field(default_factory=list)

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)


@dataclass
class RAGQuery:
    """Represents a RAG query that retrieves document chunks."""
    query_id: str
    query_tokens: int
    retrieved_chunks: List[RAGChunk]
    generation_tokens: int

    @property
    def total_context_tokens(self) -> int:
        """The total context is the user's query plus all retrieved document chunks."""
        return self.query_tokens + sum(c.token_count for c in self.retrieved_chunks)


class RAGDocumentManager:
    """Manages the ingestion and retrieval of RAG document chunks."""

    def __init__(self, cache, chunk_size: int = 512, top_k_chunks: int = 5):
        self.cache = cache # A reference to the main MultiTierCache.
        self.chunk_size = chunk_size
        self.top_k_chunks = top_k_chunks
        self.documents: Dict[str, RAGDocument] = {}
        self.chunk_index: Dict[str, RAGChunk] = {}

    def ingest_document(self, doc_id: str, total_tokens: int, model_config: ModelConfig):
        """
        Simulates the ingestion of a document.
        This involves splitting it into chunks and pre-calculating and storing the
        KV cache for each chunk in the multi-tier cache.
        """
        max_chunk_bytes = 256 * 1024**2 # Target ~256MB per chunk to limit memory pressure.
        bytes_per_token = max(model_config.kv_cache_size_per_token, 1)
        max_tokens_per_chunk = max(1, min(self.chunk_size, max_chunk_bytes // bytes_per_token))

        if max_tokens_per_chunk < self.chunk_size:
            print(f"[RAG] Adjusting chunk size for {doc_id} to {max_tokens_per_chunk} tokens "
                  f"to stay under {max_chunk_bytes / 1024**2:.0f} MB per chunk.")

        num_chunks = (total_tokens + max_tokens_per_chunk - 1) // max_tokens_per_chunk

        doc = RAGDocument(
            doc_id=doc_id,
            total_tokens=total_tokens,
            chunk_size=max_tokens_per_chunk,
            chunks=[]
        )

        for chunk_idx in range(num_chunks):
            remaining_tokens = total_tokens - chunk_idx * max_tokens_per_chunk
            chunk_tokens = min(max_tokens_per_chunk, remaining_tokens)

            chunk = RAGChunk(
                chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                doc_id=doc_id,
                chunk_index=chunk_idx,
                token_count=chunk_tokens,
                kv_cache_key=f"rag_{doc_id}_chunk_{chunk_idx}"
            )

            # Allocate and store the KV cache for this new chunk.
            try:
                success, location, write_latency = self.cache.allocate_cache(
                    key=chunk.kv_cache_key,
                    num_tokens=chunk_tokens
                )
            except MemoryError:
                print(f"[RAG] MemoryError while ingesting chunk {chunk.chunk_id}; skipping remaining chunks.")
                break
            except Exception as exc:
                print(f"[RAG] Error ingesting chunk {chunk.chunk_id}: {exc}")
                continue

            if not success:
                print(f"[RAG] Warning: Failed to allocate cache for chunk {chunk.chunk_id}.")
                continue

            chunk.storage_tier = location
            chunk.size_bytes = chunk_tokens * model_config.kv_cache_size_per_token

            doc.chunks.append(chunk)
            self.chunk_index[chunk.chunk_id] = chunk

        self.documents[doc_id] = doc
        return doc

    def retrieve_chunks(self, doc_id: str) -> List[RAGChunk]:
        """Simulates the retrieval of the top-k most relevant chunks for a query."""
        if doc_id not in self.documents:
            return []

        doc = self.documents[doc_id]

        # Simulate a realistic retrieval access pattern, where earlier chunks in a
        # document are more likely to be retrieved.
        chunk_probabilities = [1.0 / (i + 1) for i in range(len(doc.chunks))]
        total_prob = sum(chunk_probabilities)
        chunk_probabilities = [p / total_prob for p in chunk_probabilities]

        retrieved_indices = np.random.choice(
            len(doc.chunks),
            size=min(self.top_k_chunks, len(doc.chunks)),
            replace=False,
            p=chunk_probabilities
        )

        retrieved_chunks = [doc.chunks[i] for i in retrieved_indices]

        # Update access stats for the retrieved chunks.
        for chunk in retrieved_chunks:
            chunk.access_count += 1
            chunk.last_accessed = datetime.now()

        return retrieved_chunks


# ============================================================================ 
# STORAGE BACKEND CLASSES
# These classes abstract the I/O operations for each tier of the memory hierarchy.
# ============================================================================ 

class StorageBackend:
    """Abstract base class for all storage backends (GPU, CPU, NVMe)."""

    @dataclass
    class IOTiming:
        """Captures total latency along with host and device components."""
        total: float
        device: float
        host: float

    def write(self, key: str, data: np.ndarray) -> 'StorageBackend.IOTiming':
        """Writes data to the backend and returns latency breakdown."""
        raise NotImplementedError

    def read(self, key: str) -> Tuple[np.ndarray, 'StorageBackend.IOTiming']:
        """Reads data from the backend and returns the data and latency."""
        raise NotImplementedError

    def delete(self, key: str):
        """Deletes data from the backend."""
        raise NotImplementedError

    def clear(self):
        """Clears all data from the backend."""
        raise NotImplementedError


class GPUMemoryBackend(StorageBackend):
    """
    GPU VRAM storage backend.
    Uses PyTorch or CuPy for GPU operations. This is the fastest tier.
    """

    def __init__(self, use_torch=True):
        if use_torch and TORCH_AVAILABLE:
            self.backend = 'torch'
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if self.device.type == 'cpu':
                raise RuntimeError("No GPU available for PyTorch backend")
            # Pre-allocate a large chunk of GPU memory to simulate a real server environment.
            torch.cuda.set_per_process_memory_fraction(0.8, 0)
            torch.cuda.empty_cache()
        elif CUPY_AVAILABLE:
            self.backend = 'cupy'
            mempool = cp.get_default_memory_pool()
            mempool.free_all_blocks()
        else:
            raise RuntimeError("No GPU backend (PyTorch or CuPy) available.")

        self.cache = {} # Holds tensors on the GPU.
        self.pinned_memory = {} # Holds CPU memory pinned for fast async GPU transfers.

    def write(self, key: str, data: np.ndarray) -> StorageBackend.IOTiming:
        """
        Writes a NumPy array from CPU to GPU VRAM.
        Uses pinned memory and non-blocking transfers for maximum performance.
        """
        # Simple eviction mechanism if GPU runs out of memory.
        if self.backend == 'torch' and torch.cuda.is_available():
            free_memory = torch.cuda.mem_get_info()[0]
            if data.nbytes > free_memory * 0.9:
                torch.cuda.empty_cache()
                if data.nbytes > torch.cuda.mem_get_info()[0] * 0.9:
                    if len(self.cache) > 0:
                        oldest_key = list(self.cache.keys())[0]
                        del self.cache[oldest_key]
                        torch.cuda.empty_cache()

        start = time.perf_counter()

        if self.backend == 'torch':
            # Pin the CPU memory for this tensor to enable fast asynchronous transfer.
            if key not in self.pinned_memory:
                self.pinned_memory[key] = torch.from_numpy(data).pin_memory()
            # Asynchronously copy the pinned memory to the GPU.
            gpu_tensor = self.pinned_memory[key].to(self.device, non_blocking=True)
            # Wait for the transfer to complete to accurately measure latency.
            torch.cuda.synchronize()
            self.cache[key] = gpu_tensor
            del self.pinned_memory[key] # Release the pinned memory.
        else: # CuPy backend
            self.cache[key] = cp.asarray(data)
            cp.cuda.Stream.null.synchronize()

        total = time.perf_counter() - start
        # GPU transfers are all host-managed; device component equals total for now.
        return StorageBackend.IOTiming(total=total, device=total, host=total)

    def read(self, key: str) -> Tuple[np.ndarray, StorageBackend.IOTiming]:
        """Reads a tensor from GPU VRAM back to a NumPy array on the CPU."""
        if key not in self.cache:
            raise KeyError(f"Key {key} not found in GPU cache")

        start = time.perf_counter()

        if self.backend == 'torch':
            gpu_tensor = self.cache[key]
            # Asynchronously copy the tensor from GPU to CPU.
            cpu_tensor = gpu_tensor.to('cpu', non_blocking=True)
            # Wait for the transfer to complete to measure latency.
            torch.cuda.synchronize()
            data = cpu_tensor.numpy()
        else: # CuPy backend
            data = cp.asnumpy(self.cache[key])
            cp.cuda.Stream.null.synchronize()

        total = time.perf_counter() - start
        return data, StorageBackend.IOTiming(total=total, device=total, host=total)

    def delete(self, key: str):
        if key in self.cache:
            del self.cache[key]
        if key in self.pinned_memory:
            del self.pinned_memory[key]

    def clear(self):
        """Clears all tensors from the GPU cache and frees memory."""
        for key in list(self.cache.keys()):
            del self.cache[key]
        self.cache.clear()
        for key in list(self.pinned_memory.keys()):
            del self.pinned_memory[key]
        self.pinned_memory.clear()

        if self.backend == 'torch' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif self.backend == 'cupy':
            mempool = cp.get_default_memory_pool()
            pinned_mempool = cp.get_default_pinned_memory_pool()
            mempool.free_all_blocks()
            pinned_mempool.free_all_blocks()


class CPUMemoryBackend(StorageBackend):
    """CPU RAM storage backend. This is the second tier in the cache hierarchy."""

    def __init__(self):
        self.cache = {}

    def write(self, key: str, data: np.ndarray) -> StorageBackend.IOTiming:
        """Writes data by copying it into the cache dictionary."""
        start = time.perf_counter()
        self.cache[key] = np.copy(data)
        total = time.perf_counter() - start
        return StorageBackend.IOTiming(total=total, device=total, host=total)

    def read(self, key: str) -> Tuple[np.ndarray, StorageBackend.IOTiming]:
        """Reads data by copying it from the cache dictionary."""
        if key not in self.cache:
            raise KeyError(f"Key {key} not found in CPU cache")
        start = time.perf_counter()
        data = np.copy(self.cache[key])
        total = time.perf_counter() - start
        return data, StorageBackend.IOTiming(total=total, device=total, host=total)

    def delete(self, key: str):
        if key in self.cache:
            del self.cache[key]

    def clear(self):
        for key in list(self.cache.keys()):
            del self.cache[key]
        self.cache.clear()
        import gc
        gc.collect() # Force garbage collection.


class NVMeBackend(StorageBackend):
    """
    NVMe/SSD storage backend using memory-mapped files.
    This is the third and slowest tier, used for offloading from CPU RAM.
    """

    def __init__(self, base_path: str = None):
        self.temp_dir = None
        if base_path is None:
            self.temp_dir = tempfile.TemporaryDirectory(prefix="kv_cache_")
            self.base_path = Path(self.temp_dir.name)
        else:
            self.base_path = Path(base_path)
            # Ensure the cache directory exists but do not remove the mount point itself.
            if self.base_path.exists():
                if not self.base_path.is_dir():
                    raise NotADirectoryError(f"Cache path {self.base_path} exists but is not a directory.")
                # Remove only the files the benchmark generated (.npy shards).
                for entry in self.base_path.glob("*.npy"):
                    try:
                        entry.unlink()
                    except OSError:
                        pass
            else:
                self.base_path.mkdir(parents=True, exist_ok=True)

        # Final sanity check.
        if not self.base_path.exists():
            raise OSError(f"Cache directory {self.base_path} does not exist and could not be created.")

        self.metadata = {}

    def _get_path(self, key: str) -> Path:
        """Constructs the file path for a given cache key."""
        return self.base_path / f"{key}.npy"

    def write(self, key: str, data: np.ndarray) -> StorageBackend.IOTiming:
        """Writes a NumPy array to a binary .npy file on disk."""
        start = time.perf_counter()
        path = self._get_path(key)

        with open(path, 'wb') as f:
            np.save(f, data, allow_pickle=False)
            # Host serialization (NumPy header + buffer copy) completes here.
            post_save = time.perf_counter()
            f.flush()
            # fsync blocks until the kernel persists data to the device.
            os.fsync(f.fileno())
            post_fsync = time.perf_counter()

        self.metadata[key] = {'shape': data.shape, 'dtype': str(data.dtype), 'size': data.nbytes}

        host_time = post_save - start
        device_time = post_fsync - post_save
        total = post_fsync - start
        return StorageBackend.IOTiming(total=total, device=device_time, host=host_time)

    def read(self, key: str) -> Tuple[np.ndarray, StorageBackend.IOTiming]:
        """
        Reads a .npy file from disk.

        IMPORTANT: This method is designed to force actual disk I/O for accurate storage
        benchmarking. It uses posix_fadvise() to drop the file from the Linux page cache
        before reading, ensuring that:
        1. Every read operation hits the physical storage device (NVMe/SSD)
        2. iostat and other system monitoring tools accurately reflect storage I/O
        3. Latency measurements represent real-world storage performance

        Without this, Linux would serve reads from the page cache, making it appear as if
        no disk I/O is occurring (iostat shows 0 r/s), which defeats the purpose of a
        storage benchmark.
        """
        start = time.perf_counter()
        path = self._get_path(key)

        if not path.exists():
            raise KeyError(f"Key {key} not found in NVMe cache")

        # CRITICAL FIX: Drop this file from the Linux page cache before reading.
        # This ensures that the subsequent read operation will be served from the actual
        # storage device rather than from cached memory.
        try:
            fd = os.open(path, os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, 4)  # POSIX_FADV_DONTNEED
            except AttributeError:
                pass
            finally:
                os.close(fd)
        except Exception:
            pass

        pre_load = time.perf_counter()
        data = np.load(path, allow_pickle=False)
        load_done = time.perf_counter()
        # Convert to a standard numpy array to ensure the full data is loaded into memory.
        data = np.array(data)
        copy_done = time.perf_counter()

        device_time = load_done - pre_load
        host_time = (pre_load - start) + (copy_done - load_done)
        total = copy_done - start
        return data, StorageBackend.IOTiming(total=total, device=device_time, host=host_time)

    def delete(self, key: str):
        path = self._get_path(key)
        if path.exists():
            path.unlink()
        if key in self.metadata:
            del self.metadata[key]

    def clear(self):
        """Deletes all .npy files from the cache directory."""
        for file in self.base_path.glob("*.npy"):
            file.unlink()
        self.metadata.clear()

    def __del__(self):
        """Cleans up the temporary directory when the object is destroyed."""
        if self.temp_dir:
            import shutil
            shutil.rmtree(self.temp_dir, ignore_errors=True)


class KVCacheGenerator:
    """Generates realistic-looking KV cache data for testing."""

    def __init__(self, model_config: ModelConfig, global_seed: Optional[int] = None):
        self.model_config = model_config
        self.global_seed = 0 if global_seed is None else int(global_seed)

    def _seed_from_key(self, key: str) -> int:
        # Use stable cryptographic hash to get deterministic 64-bit seed
        h = hashlib.sha256(key.encode('utf-8')).digest()
        key_hash64 = int.from_bytes(h[:8], 'little')
        return (key_hash64 ^ self.global_seed) & 0xFFFFFFFFFFFFFFFF

    def generate(self, sequence_length: int, key: Optional[str] = None) -> np.ndarray:
        """
        Generates a NumPy array with the correct shape and dtype for a KV cache.
        The data itself is random noise, but is generated deterministically if a key is provided.
        """
        # The shape of a KV cache tensor is typically:
        # (num_layers, 2 (for K/V), sequence_length, num_kv_heads, head_dimension)
        kv_shape = (
            self.model_config.num_layers,
            2,  # K and V
            sequence_length,
            self.model_config.kv_heads,
            self.model_config.kv_dim_per_head
        )

        dtype = np.float16 if 'float16' in self.model_config.dtype else np.float32
        
        if key is None:
            # Fallback to global RNG if no key is provided (less deterministic in multithreading)
            rng = np.random.default_rng(self.global_seed)
        else:
            # Generate a seed deterministically from the key and global seed
            seed = self._seed_from_key(key)
            rng = np.random.default_rng(seed & 0xFFFFFFFF)

        data = rng.uniform(-1.0, 1.0, size=kv_shape).astype(dtype)
        return data


# ============================================================================ 
# ENHANCED MULTI-TIER CACHE
# This is the core logic of the benchmark, managing the three-tier hierarchy.
# ============================================================================ 

class MultiTierCache:
    """
    Manages KV cache data across GPU, CPU, and NVMe tiers.

    This class is the heart of the benchmark. It orchestrates where cache data is
    written to and read from based on available space and access patterns.
    It is heavily instrumented to collect detailed performance metrics.
    """

    def __init__(self,
                 model_config: ModelConfig,
                 gpu_memory_gb: float,
                 cpu_memory_gb: float,
                 cache_dir: str = None,
                 eviction_policy: str = 'lru',
                 performance_profile: str = 'latency',
                 seed: Optional[int] = None):

        self.model_config = model_config
        self.gpu_memory_limit = gpu_memory_gb * 1024**3
        self.cpu_memory_limit = cpu_memory_gb * 1024**3
        self.eviction_policy = eviction_policy
        self.performance_profile = performance_profile
        self.seed = seed

        # Initialize storage backends for each tier.
        self.backends = {}
        try:
            if TORCH_AVAILABLE or CUPY_AVAILABLE:
                self.backends['gpu'] = GPUMemoryBackend(use_torch=TORCH_AVAILABLE)
        except Exception as e:
            print(f"Warning: Could not initialize GPU backend: {e}")

        self.backends['cpu'] = CPUMemoryBackend()
        self.backends['nvme'] = NVMeBackend(base_path=cache_dir)

        self.generator = KVCacheGenerator(model_config, global_seed=self.seed)

        # Metadata tracking for all cache entries across all tiers.
        self.cache_entries = {} # Main dictionary mapping a key to its metadata.
        self.entry_locks: Dict[str, threading.Lock] = {} # Fine-grained locks per cache key.
        self.gpu_memory_used = 0
        self.cpu_memory_used = 0

        # Global locks for managing shared state.
        self.metadata_lock = threading.Lock()  # For coarse-grained operations on the cache_entries dict itself.
        self.memory_lock = threading.Lock()     # For updating the gpu_memory_used and cpu_memory_used counters.
        self.stats_lock = threading.Lock()      # For updating the performance statistics dictionary.

        # Dictionary for collecting a wide range of performance metrics.
        self.stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'evictions': 0,
            'offloads_cpu': 0, # Prefills that went directly to CPU.
            'offloads_nvme': 0, # Prefills that went directly to NVMe.

            # Latency lists for each tier and operation.
            'gpu_read_latencies': [], 'cpu_read_latencies': [], 'nvme_read_latencies': [],
            'gpu_write_latencies': [], 'cpu_write_latencies': [], 'nvme_write_latencies': [],
            'nvme_read_device_latencies': [], 'nvme_read_host_latencies': [],
            'nvme_write_device_latencies': [], 'nvme_write_host_latencies': [],

            # Phase-specific I/O metrics.
            'prefill_writes': 0, 'decode_reads': 0,
            'prefill_bytes_written': 0, 'decode_bytes_read': 0,

            # Cache type metrics for analyzing hit sources.
            'system_prompt_hits': 0, 'common_phrase_hits': 0,
            'user_cache_hits': 0, 'multi_turn_hits': 0,

            # Aggregate I/O metrics.
            'total_read_bytes': 0, 'total_write_bytes': 0,
            'read_operations': 0, 'write_operations': 0,

            # New counter for NVMe tokens processed (for throughput assessment)
            'nvme_tokens_processed': 0,
        }

    def _get_entry_lock(self, key: str) -> threading.Lock:
        """Get or create a lock for a specific cache entry to ensure thread safety."""
        with self.metadata_lock:
            if key not in self.entry_locks:
                self.entry_locks[key] = threading.Lock()
            return self.entry_locks[key]

    def allocate_cache(self, key: str, num_tokens: int, phase: InferencePhase = InferencePhase.PREFILL) -> Tuple[bool, str, float]:
        """
        Allocates and writes a new KV cache entry to the most appropriate tier.
        This simulates the 'prefill' phase.

        Args:
            key: The unique key for the cache entry.
            num_tokens: The number of tokens to generate cache for.
            phase: The current inference phase (should be PREFILL).

        Returns:
            A tuple of (success_boolean, location_string, write_latency_seconds).
        """
        # Quick check to see if the key already exists to avoid redundant work.
        with self.metadata_lock:
            if key in self.cache_entries:
                return True, self.cache_entries[key]['location'], 0.0

        # Generate the KV cache data. This is computationally expensive and done outside locks.
        try:
            data = self.generator.generate(sequence_length=num_tokens, key=key)
        except MemoryError:
            print(f"[KVCache] MemoryError generating cache for key {key} ({num_tokens} tokens)")
            return False, 'none', 0.0
        except Exception as exc:
            print(f"[KVCache] Failed to generate cache for key {key}: {exc}")
            return False, 'none', 0.0

        size_bytes = data.nbytes

        # Update write statistics.
        with self.stats_lock:
            if phase == InferencePhase.PREFILL:
                self.stats['prefill_writes'] += 1
                self.stats['prefill_bytes_written'] += size_bytes
            self.stats['write_operations'] += 1
            self.stats['total_write_bytes'] += size_bytes

        # --- Tiering Logic ---
        # Decide which tier to write to based on available memory.
        with self.memory_lock:
            # Tier 1: GPU. Check if there's space in the GPU budget (with a 20% buffer).
            if 'gpu' in self.backends and self.gpu_memory_used + size_bytes < self.gpu_memory_limit * 0.8:
                self.gpu_memory_used += size_bytes
                allocated_tier = 'gpu'
            # Tier 2: CPU. Check if there's space in the CPU budget.
            elif self.cpu_memory_used + size_bytes < self.cpu_memory_limit * 0.8:
                self.cpu_memory_used += size_bytes
                allocated_tier = 'cpu'
            # Tier 3: NVMe. If no space in RAM, offload to disk.
            else:
                allocated_tier = 'nvme'

        # Perform the actual write operation to the chosen backend.
        try:
            if allocated_tier == 'gpu':
                timing = self.backends['gpu'].write(key, data)
            elif allocated_tier == 'cpu':
                timing = self.backends['cpu'].write(key, data)
            else:
                timing = self.backends['nvme'].write(key, data)

            # After a successful write, update the central metadata dictionary.
            with self.metadata_lock:
                self.cache_entries[key] = {
                    'location': allocated_tier,
                    'size': size_bytes,
                    'last_access': time.time(),
                    'access_count': 1
                }

            # Record latency and offload stats.
            with self.stats_lock:
                if allocated_tier == 'cpu':
                    self.stats['offloads_cpu'] += 1
                    self.stats['cpu_write_latencies'].append(timing.total)
                elif allocated_tier == 'nvme':
                    self.stats['offloads_nvme'] += 1
                    self.stats['nvme_write_latencies'].append(timing.total)
                    self.stats['nvme_write_device_latencies'].append(timing.device)
                    self.stats['nvme_write_host_latencies'].append(timing.host)
                    self.stats['nvme_tokens_processed'] += num_tokens
                elif allocated_tier == 'gpu':
                    self.stats['gpu_write_latencies'].append(timing.total)

            del data # Free the memory for the generated data.
            return True, allocated_tier, timing.total

        except Exception as e:
            # If the write fails, roll back the memory reservation.
            with self.memory_lock:
                if allocated_tier == 'gpu':
                    self.gpu_memory_used -= size_bytes
                elif allocated_tier == 'cpu':
                    self.cpu_memory_used -= size_bytes
            del data
            return False, 'none', 0.0

    def access_cache(self, key: str, phase: InferencePhase = InferencePhase.DECODE,
                     cache_type: str = 'user') -> Tuple[Optional[str], float]:
        """
        Accesses an existing cached entry and records the read performance.
        This simulates the 'decode' phase.

        Args:
            key: The unique key for the cache entry to access.
            phase: The current inference phase (should be DECODE).
            cache_type: The type of cache being accessed (for detailed stats).

        Returns:
            A tuple of (location_string, read_latency_seconds).
        """
        # First, check if the metadata for the key exists.
        with self.metadata_lock:
            if key not in self.cache_entries:
                with self.stats_lock:
                    self.stats['cache_misses'] += 1
                return None, 0.0

            entry = self.cache_entries[key]
            location = entry['location']
            entry_size = entry['size']

        # Get the specific lock for this key to handle concurrent access.
        entry_lock = self._get_entry_lock(key)

        with entry_lock:
            # Update metadata (access time, count) and performance stats.
            with self.metadata_lock:
                entry = self.cache_entries[key]
                entry['last_access'] = time.time()
                entry['access_count'] += 1

            with self.stats_lock:
                self.stats['cache_hits'] += 1

                # Track hits by cache type for deeper analysis.
                if cache_type == 'system': self.stats['system_prompt_hits'] += 1
                elif cache_type == 'common': self.stats['common_phrase_hits'] += 1
                elif cache_type == 'multi_turn': self.stats['multi_turn_hits'] += 1
                else: self.stats['user_cache_hits'] += 1

                # Track phase-specific I/O.
                if phase == InferencePhase.DECODE:
                    self.stats['decode_reads'] += 1
                    self.stats['decode_bytes_read'] += entry_size

                self.stats['read_operations'] += 1
                self.stats['total_read_bytes'] += entry_size

            # Perform the actual read from the correct backend (GPU, CPU, or NVMe).
            try:
                _, timing = self.backends[location].read(key)

                # Record the latency for the specific tier that was read from.
                with self.stats_lock:
                    if location == 'gpu':
                        self.stats['gpu_read_latencies'].append(timing.total)
                    elif location == 'cpu':
                        self.stats['cpu_read_latencies'].append(timing.total)
                    else:
                        self.stats['nvme_read_latencies'].append(timing.total)
                        self.stats['nvme_read_device_latencies'].append(timing.device)
                        self.stats['nvme_read_host_latencies'].append(timing.host)
                        
                        #The access_cache function already retrieves the size of the entry in bytes: entry_size = entry['size'].
                        #The number of tokens can be calculated by dividing entry_size by the size of a single token's KV cache, which is available via self.model_config.kv_cache_size_per_token.
                        #This calculation should happen only when the read is from the 'nvme' tier.
                        if self.model_config.kv_cache_size_per_token > 0:
                            num_tokens = entry_size / self.model_config.kv_cache_size_per_token
                            self.stats['nvme_tokens_processed'] += num_tokens

                return location, timing.total
            except Exception as e:
                # In case of a read error, return the location but with zero latency.
                return location, 0.0

    def _evaluate_storage_performance(self, duration: float) -> Dict:
        """
        Evaluates storage performance against pre-defined MLPerf Storage WG criteria.
        This provides a clear PASS/FAIL assessment of the storage system.
        """
        criteria = []
        all_passed = True

        # Throughput-focused profile for MLPerf submission
        if self.performance_profile == 'throughput':
            # Criterion: Throughput should be based on tokens processed by the NVMe tier.
            nvme_tokens = self.stats.get('nvme_tokens_processed', 0)
            # Correctly use the benchmark's full duration for an accurate tok/s calculation.
            throughput = nvme_tokens / duration if duration > 0 else 0
            
            passed = throughput > 0  # Simple check to ensure it ran
            criteria.append({
                'name': 'Throughput (tok/s)',
                'target': '>0', 'actual': f"{throughput:.2f}", 'unit': 'tok/s', 'passed': passed
            })
            all_passed = all_passed and passed
            
            return {
                'overall_status': 'PASS' if all_passed else 'FAIL',
                'criteria': criteria,
                'passed_count': sum(1 for c in criteria if c['passed']),
                'total_count': len(criteria)
            }

        # Latency-focused profile (default)
        # Criterion 1: NVMe Write P95 latency should be less than 500ms.
        nvme_write_device = self.stats.get('nvme_write_device_latencies', [])
        nvme_write_total = self.stats.get('nvme_write_latencies', [])
        nvme_write_basis = nvme_write_device if nvme_write_device else nvme_write_total
        if nvme_write_basis:
            nvme_write_p95 = np.percentile(nvme_write_basis, 95) * 1000
            passed = nvme_write_p95 < 500
            criteria.append({
                'name': 'NVMe Write P95 < 500ms',
                'target': 500, 'actual': nvme_write_p95, 'unit': 'ms', 'passed': passed
            })
            all_passed = all_passed and passed

        # Criterion 2: NVMe Read P95 latency should be less than 200ms.
        nvme_read_device = self.stats.get('nvme_read_device_latencies', [])
        nvme_read_total = self.stats.get('nvme_read_latencies', [])
        nvme_read_basis = nvme_read_device if nvme_read_device else nvme_read_total
        if nvme_read_basis:
            nvme_read_p95 = np.percentile(nvme_read_basis, 95) * 1000
            passed = nvme_read_p95 < 200
            criteria.append({
                'name': 'NVMe Read P95 < 200ms',
                'target': 200, 'actual': nvme_read_p95, 'unit': 'ms', 'passed': passed
            })
            all_passed = all_passed and passed

        # Criterion 3: CPU RAM P95 latency should be less than 150ms.
        # This accounts for large memory copies within RAM.
        cpu_read_lats = self.stats.get('cpu_read_latencies', [])
        cpu_write_lats = self.stats.get('cpu_write_latencies', [])
        if cpu_read_lats or cpu_write_lats:
            all_cpu_lats = cpu_read_lats + cpu_write_lats
            cpu_p95 = np.percentile(all_cpu_lats, 95) * 1000
            passed = cpu_p95 < 150
            criteria.append({
                'name': 'CPU RAM P95 < 150ms',
                'target': 150, 'actual': cpu_p95, 'unit': 'ms', 'passed': passed
            })
            all_passed = all_passed and passed

        # Criterion 4: Overall cache hit rate should be above 30% for a realistic workload.
        total_accesses = self.stats['cache_hits'] + self.stats['cache_misses']
        if total_accesses > 0:
            hit_rate = self.stats['cache_hits'] / total_accesses
            passed = hit_rate > 0.3
            criteria.append({
                'name': 'Cache Hit Rate > 30%',
                'target': 0.3, 'actual': hit_rate, 'unit': 'ratio', 'passed': passed
            })
            all_passed = all_passed and passed

        return {
            'overall_status': 'PASS' if all_passed else 'FAIL',
            'criteria': criteria,
            'passed_count': sum(1 for c in criteria if c['passed']),
            'total_count': len(criteria)
        }

    def get_stats(self, duration: float) -> Dict:
        """Gathers and returns a comprehensive dictionary of all performance statistics."""
        # Snapshot stats and metadata under locks to ensure consistency.
        with self.stats_lock:
            total_accesses = self.stats['cache_hits'] + self.stats['cache_misses']
            hit_rate = self.stats['cache_hits'] / total_accesses if total_accesses > 0 else 0
            stats_snapshot = self.stats.copy()

        with self.metadata_lock:
            gpu_entries = sum(1 for e in self.cache_entries.values() if e['location'] == 'gpu')
            cpu_entries = sum(1 for e in self.cache_entries.values() if e['location'] == 'cpu')
            nvme_entries = sum(1 for e in self.cache_entries.values() if e['location'] == 'nvme')

        with self.memory_lock:
            gpu_mem_used = self.gpu_memory_used
            cpu_mem_used = self.cpu_memory_used

        # Get the pass/fail assessment.
        storage_health = self._evaluate_storage_performance(duration)

        stats = {
            'cache_hit_rate': hit_rate,
            'cache_hits': stats_snapshot['cache_hits'],
            'cache_misses': stats_snapshot['cache_misses'],
            'gpu_entries': gpu_entries,
            'cpu_entries': cpu_entries,
            'nvme_entries': nvme_entries,
            'gpu_memory_used_gb': gpu_mem_used / 1024**3,
            'cpu_memory_used_gb': cpu_mem_used / 1024**3,
            'offloads_cpu': stats_snapshot['offloads_cpu'],
            'offloads_nvme': stats_snapshot['offloads_nvme'],
            'storage_health': storage_health,
            'prefill_writes': self.stats['prefill_writes'],
            'decode_reads': self.stats['decode_reads'],
            'prefill_bytes_written_gb': self.stats['prefill_bytes_written'] / 1024**3,
            'decode_bytes_read_gb': self.stats['decode_bytes_read'] / 1024**3,
            'system_prompt_hits': self.stats['system_prompt_hits'],
            'common_phrase_hits': self.stats['common_phrase_hits'],
            'user_cache_hits': self.stats['user_cache_hits'],
            'multi_turn_hits': self.stats['multi_turn_hits'],
            'total_read_bytes': self.stats['total_read_bytes'],
            'total_write_bytes': self.stats['total_write_bytes'],
            'total_read_gb': self.stats['total_read_bytes'] / 1024**3,
            'total_write_gb': self.stats['total_write_bytes'] / 1024**3,
            'read_write_ratio': self.stats['total_read_bytes'] / max(self.stats['total_write_bytes'], 1),
            'read_iops': self.stats['read_operations'],
            'write_iops': self.stats['write_operations'],
        }

        # Add latency percentiles for each tier.
        for tier in ['gpu', 'cpu', 'nvme']:
            for op in ['read', 'write']:
                latencies = self.stats[f'{tier}_{op}_latencies']
                if latencies:
                    lat_array = np.array(latencies)
                    stats[f'{tier}_{op}_p50_ms'] = np.percentile(lat_array, 50) * 1000
                    stats[f'{tier}_{op}_p95_ms'] = np.percentile(lat_array, 95) * 1000
                    stats[f'{tier}_{op}_p99_ms'] = np.percentile(lat_array, 99) * 1000

        # Expose NVMe latency component breakdowns when present.
        for op in ['read', 'write']:
            device_latencies = self.stats[f'nvme_{op}_device_latencies']
            host_latencies = self.stats[f'nvme_{op}_host_latencies']
            if device_latencies:
                device_array = np.array(device_latencies)
                stats[f'nvme_{op}_device_p50_ms'] = np.percentile(device_array, 50) * 1000
                stats[f'nvme_{op}_device_p95_ms'] = np.percentile(device_array, 95) * 1000
                stats[f'nvme_{op}_device_p99_ms'] = np.percentile(device_array, 99) * 1000
            if host_latencies:
                host_array = np.array(host_latencies)
                stats[f'nvme_{op}_host_p50_ms'] = np.percentile(host_array, 50) * 1000
                stats[f'nvme_{op}_host_p95_ms'] = np.percentile(host_array, 95) * 1000
                stats[f'nvme_{op}_host_p99_ms'] = np.percentile(host_array, 99) * 1000

        return stats


# ============================================================================ 
# FEATURE 5: ADAPTIVE AUTOSCALING
# Automatically adjusts the user load to find a performance limit.
# ============================================================================ 

@dataclass
class StorageMetrics:
    """A snapshot of storage performance metrics at a point in time."""
    timestamp: float
    read_throughput_gbps: float
    write_throughput_gbps: float
    read_iops: int
    write_iops: int
    read_latency_p95_ms: float
    write_latency_p95_ms: float
    queue_depth: int
    is_saturated: bool = False
    saturation_level: float = 0.0
   

    # @property
    # def is_saturated(self) -> bool:
    #     """Determines if storage is saturated based on latency and queue depth thresholds."""
    #     return (
    #         self.read_latency_p95_ms > 100 or
    #         self.write_latency_p95_ms > 50 or
    #         self.queue_depth > 100
    #     )


class StorageMonitor:
    """Monitors storage performance in real-time to feed the autoscaler."""

    def __init__(self, benchmark_instance, sampling_interval_ms: float = 100):
        self.benchmark_instance = benchmark_instance
        self.sampling_interval = sampling_interval_ms / 1000.0
        self.last_collection_time = None
        self.last_total_read = 0
        self.last_total_write = 0
        self.metrics_history = []
        self.lock = threading.Lock()

    def collect_metrics(self, cache, queue_size):
        """Collects all relevant performance metrics."""
        now = time.time()
        if self.last_collection_time is None:
            self.last_collection_time = now
            self.last_total_read = cache.stats.get('total_read_bytes', 0)
            self.last_total_write = cache.stats.get('total_write_bytes', 0)
            return {}

        elapsed = now - self.last_collection_time
        if elapsed == 0:
            return {}

        # The duration for get_stats should be the total benchmark duration, not the interval
        stats = cache.get_stats(duration=self.benchmark_instance.duration)
        current_total_read = stats.get('total_read_bytes', 0)
        current_total_write = stats.get('total_write_bytes', 0)

        # Calculate deltas since the last sample
        read_delta = max(current_total_read - self.last_total_read, 0)
        write_delta = max(current_total_write - self.last_total_write, 0)

        # Calculate read and write throughput in GB/s
        read_throughput = (read_delta / 1024**3) / elapsed
        write_throughput = (write_delta / 1024**3) / elapsed

        # Calculate queue depth as the number of requests in the queue
        queue_depth = queue_size

        # Estimate read and write IOPS based on common block sizes (4KB for reads, 16KB for writes)
        read_iops = int((read_delta / 4096) / elapsed) if elapsed > 0 else 0
        write_iops = int((write_delta / (16 * 1024)) / elapsed) if elapsed > 0 else 0

        # Default to 0.0 if the keys don't exist (e.g., at the start of the run).
        read_latency_p95_ms = stats.get('nvme_read_p95_ms', 0.0)
        write_latency_p95_ms = stats.get('nvme_write_p95_ms', 0.0)

        # --- Saturation Detection Logic ---
        is_saturated = False
        if len(self.metrics_history) >= 2:
            # Compare with the previous metric
            prev_metric = self.metrics_history[-2]
            if (prev_metric.read_latency_p95_ms < 100 and prev_metric.write_latency_p95_ms < 50 and prev_metric.queue_depth < 100):
                # If the previous metric was not saturated, check for a sudden increase in latency or queue depth
                if (abs(prev_metric.read_latency_p95_ms - read_latency_p95_ms) > 20 or
                    abs(prev_metric.write_latency_p95_ms - write_latency_p95_ms) > 10 or
                    abs(prev_metric.queue_depth - queue_depth) > 10):
                    is_saturated = True
            else:
                # If the previous metric was saturated, check if it's still above the thresholds
                if (read_latency_p95_ms > 120 or write_latency_p95_ms > 60 or queue_depth > 120):
                    is_saturated = True

        # Create a new StorageMetrics object for this sample
        metrics = StorageMetrics(
            timestamp=now,
            read_throughput_gbps=read_throughput,
            write_throughput_gbps=write_throughput,
            read_iops=read_iops,
            write_iops=write_iops,
            read_latency_p95_ms=read_latency_p95_ms,
            write_latency_p95_ms=write_latency_p95_ms,
            queue_depth=queue_depth,
            is_saturated=is_saturated
        )

        # Add to the history and calculate saturation using a snapshot for thread safety.
        with self.lock:
            self.metrics_history.append(metrics)
            saturation_level = self._compute_saturation_from_history(self.metrics_history)

        metrics.saturation_level = saturation_level

        # Update baselines for the next interval.
        self.last_collection_time = now
        self.last_total_read = current_total_read
        self.last_total_write = current_total_write
        return metrics

    def get_saturation_level(self) -> float:
        """
        Calculates the storage saturation level (0.0 = idle, 1.0 = saturated).
        Uses heuristics like increasing latency and plateauing throughput.
        """
        with self.lock:
            history_snapshot = list(self.metrics_history)

        return self._compute_saturation_from_history(history_snapshot)

    def _compute_saturation_from_history(self, history: List[StorageMetrics]) -> float:
        if len(history) < 10:
            return 0.0

        recent_metrics = history[-10:]

        # Check if latency is trending upwards.
        latencies = [m.read_latency_p95_ms for m in recent_metrics]
        if len(latencies) > 1:
            latency_trend = np.polyfit(range(len(latencies)), latencies, 1)[0]
        else:
            latency_trend = 0

        # Check if throughput is plateauing (low variance).
        throughputs = [m.read_throughput_gbps + m.write_throughput_gbps for m in recent_metrics]
        throughput_variance = np.std(throughputs) / (np.mean(throughputs) + 0.01)

        # Combine indicators to get a single saturation score.
        latency_factor = min(max(latencies) / 100, 1.0)
        plateau_factor = 1.0 if throughput_variance < 0.1 and latency_trend > 0 else 0.5

        saturation = latency_factor * plateau_factor
        return min(saturation, 1.0)


class WorkloadAutoscaler:
    """Automatically scales the number of simulated users to find a performance limit."""

    def __init__(self,
                 mode: str = 'qos',
                 initial_users: int = 10,
                 target_saturation: float = 0.8,
                 scale_interval_seconds: int = 10):
        self.mode = mode
        self.current_users = initial_users
        self.target_saturation = target_saturation
        self.scale_interval = scale_interval_seconds
        self.min_users = 1
        self.max_users = 10000
        self.scaling_history = []
        self.lock = threading.Lock()
        
        # State for 'qos' mode (latency-driven)
        self.cooldown_counter = 0
        self.cooldown_period = 3 # Wait for 3 cycles after a scale-down action
        self.downward_trend_count = 0

        # State for 'capacity' mode (throughput-driven)
        self.capacity_stage = 0
        self.last_throughput = 0.0
        self.peak_throughput = 0.0
        self.peak_user_count = 0
        self.capacity_test_finished = False
        self.throughput_history: List[float] = []
        # Clip capacity-mode step ramps so we do not overwhelm the system in a single jump.
        self.capacity_initial_fraction = 0.4
        self.capacity_scale_fraction = 0.2
        self.capacity_min_step = 5
        self.capacity_max_step = 100

    def calculate_scale_action(
        self,
        metrics: Optional[StorageMetrics],
        current_throughput: float,
        saturation_level: Optional[float] = None
    ) -> Tuple[str, int]:
        """Decides the next scaling action based on the selected mode."""
        if self.mode == 'qos':
            if not metrics: return 'stable', self.current_users
            return self._calculate_qos_action(metrics, saturation_level)
        elif self.mode == 'capacity':
            return self._calculate_capacity_action(current_throughput)
        return 'stable', self.current_users

    def _calculate_qos_action(self, metrics: StorageMetrics, saturation_level: Optional[float]) -> Tuple[str, int]:
        """Determines the scaling action for 'qos' mode based on latency and saturation."""
        with self.lock:
            if self.cooldown_counter > 0:
                self.cooldown_counter -= 1
                return 'hold', self.current_users # In cooldown from a recent scale-down

            saturation = saturation_level
            if saturation is None:
                saturation = 1.0 if metrics.is_saturated else 0.0

            action = 'hold'
            target_users = self.current_users

            if saturation > self.target_saturation * 1.1: # Significantly over target
                self.downward_trend_count += 1
                if self.downward_trend_count >= 2: # Consistently over target
                    target_users = max(int(self.current_users * 0.8), self.min_users)
                    if target_users < self.current_users:
                        self.current_users = target_users
                        self.cooldown_counter = self.cooldown_period
                        action = 'scale_down'
            elif saturation < self.target_saturation * 0.9: # Significantly under target
                self.downward_trend_count = 0
                target_users = min(int(self.current_users * 1.2), self.max_users)
                if target_users > self.current_users:
                    self.current_users = target_users
                    action = 'scale_up'
            else: # Within target range
                self.downward_trend_count = 0

            return action, self.current_users
        return 'hold', self.current_users

    def _calculate_capacity_action(self, current_throughput: float) -> Tuple[str, int]:
        """
        Determines the scaling action for 'capacity' mode.
        Aggressively adds users until throughput stops increasing.
        """
        with self.lock:
            self.throughput_history.append(current_throughput)

            if not self.throughput_history or len(self.throughput_history) == 1:
                # First datapoint: kick off with a moderate scale-up to start discovery
                self.peak_throughput = current_throughput
                self.peak_user_count = self.current_users
                step = self._compute_capacity_step(self.capacity_initial_fraction)
                new_users = min(self.current_users + step, self.max_users)
                if new_users > self.current_users:
                    self.current_users = new_users
                    return 'scale_up', self.current_users
                return 'hold', self.current_users

            if current_throughput > self.peak_throughput * 1.01: # Require >1% increase
                self.peak_throughput = current_throughput
                self.peak_user_count = self.current_users
                self.downward_trend_count = 0
                step = self._compute_capacity_step(self.capacity_scale_fraction)
                new_users = min(self.current_users + step, self.max_users)
                if new_users > self.current_users:
                    self.current_users = new_users
                    return 'scale_up', self.current_users
                return 'hold', self.current_users

            self.downward_trend_count += 1
            if self.downward_trend_count >= 2:
                self.capacity_test_finished = True
                print(f"INFO: Peak capacity found at {self.peak_throughput:.2f} tok/s. Stopping test.")
                return 'stop', self.current_users

            return 'hold', self.current_users
        return 'hold', self.current_users

    def _compute_capacity_step(self, fraction: float) -> int:
        """Calculate a bounded capacity-mode step for smoother scaling."""
        raw_step = max(int(self.current_users * fraction), self.capacity_min_step)
        return min(raw_step, self.capacity_max_step)


# ============================================================================ 
# FEATURE 7: QOS MONITORING
# Tracks QoS compliance for different user priority levels.
# ============================================================================ 

class QoSMonitor:
    """Monitors and reports on QoS compliance in real-time."""

    def __init__(self):
        self.requests_by_qos: Dict[QoSLevel, List[InferenceRequest]] = {level: [] for level in QoSLevel}
        self.lock = threading.Lock()
        self.violations_by_qos: Dict[QoSLevel, int] = {level: 0 for level in QoSLevel}

    def record_request(self, request: InferenceRequest):
        """Records a completed request and checks if it violated its SLA."""
        with self.lock:
            self.requests_by_qos[request.qos_level].append(request)

            # Check for SLA violation.
            sla = QOS_PROFILES[request.qos_level]
            if request.total_latency_ms > sla.target_latency_p95_ms:
                self.violations_by_qos[request.qos_level] += 1
                sla.violations += 1
            sla.total_requests += 1

    def get_qos_metrics(self, qos_level: QoSLevel) -> Dict:
        """Gets performance metrics for a specific QoS level."""
        with self.lock:
            requests = self.requests_by_qos[qos_level]
            if not requests: return {'no_data': True}

            latencies = [r.total_latency_ms for r in requests]
            sla = QOS_PROFILES[qos_level]

            return {
                'total_requests': len(requests),
                'latency_ms': {
                    'mean': np.mean(latencies), 'p50': np.percentile(latencies, 50),
                    'p95': np.percentile(latencies, 95), 'p99': np.percentile(latencies, 99),
                    'max': np.max(latencies),
                },
                'sla': {
                    'target_p95_ms': sla.target_latency_p95_ms,
                    'actual_p95_ms': np.percentile(latencies, 95),
                    'compliance': sla.sla_compliance,
                    'met': sla.sla_compliance >= 0.95

                }
            }

    def get_all_qos_metrics(self) -> Dict:
        """Gets metrics for all QoS levels."""
        return {level.value: self.get_qos_metrics(level) for level in QoSLevel}


# ============================================================================ 
# FEATURE 6: TRACE-DRIVEN VALIDATION
# Validates the benchmark's accuracy by comparing its results to a real trace.
# ============================================================================ 

@dataclass
class RealTraceEntry:
    """Represents a single entry from a real-world LLM inference trace file."""
    timestamp: float
    request_id: str
    user_id: str
    context_tokens: int
    generation_tokens: int
    phase: str
    cache_hit: bool
    cache_tier: str
    read_bytes: int
    write_bytes: int
    read_latency_ms: float
    write_latency_ms: float
    model_name: str
    conversation_id: Optional[str] = None
    turn_number: Optional[int] = None
    prefix_cached: bool = False


class ValidationEngine:
    """Validates benchmark accuracy against real-world traces."""

    def __init__(self, trace_path: Optional[str] = None):
        self.trace_path = trace_path
        self.trace_stats = None

    def load_trace(self) -> Dict:
        """Loads and analyzes a trace file, or returns synthetic stats if none provided."""
        if not self.trace_path or not os.path.exists(self.trace_path):
            # Return synthetic trace stats for testing purposes.
            return {
                'total_requests': 1000, 'duration_seconds': 100, 'cache_hit_rate': 0.65,
                'read_write_ratio': 10.0, 'context_tokens_mean': 1024, 'generation_tokens_mean': 200,
            }

        with open(self.trace_path, 'r') as f:
            data = json.load(f)
            entries = [RealTraceEntry(**entry) for entry in data]

        # Calculate key statistics from the real trace.
        self.trace_stats = {
            'total_requests': len(entries),
            'cache_hit_rate': sum(1 for e in entries if e.cache_hit) / len(entries),
            'read_write_ratio': sum(e.read_bytes for e in entries) / max(sum(e.write_bytes for e in entries), 1),
            'context_tokens_mean': np.mean([e.context_tokens for e in entries]),
            'generation_tokens_mean': np.mean([e.generation_tokens for e in entries]),
        }
        return self.trace_stats

    def validate_benchmark(self, benchmark_results: Dict) -> Dict:
        """Compares key benchmark results against the trace to calculate an error percentage."""
        if self.trace_stats is None:
            self.trace_stats = self.load_trace()

        summary = benchmark_results.get('summary', {})
        cache_stats = summary.get('cache_stats', {})
        comparison = {}

        # Compare cache hit rate.
        bench_hit_rate = cache_stats.get('cache_hit_rate', 0)
        trace_hit_rate = self.trace_stats['cache_hit_rate']
        hit_rate_error = abs(bench_hit_rate - trace_hit_rate) / trace_hit_rate * 100

        comparison['cache_hit_rate'] = {
            'benchmark': bench_hit_rate, 'trace': trace_hit_rate,
            'error_pct': hit_rate_error, 'within_5pct': hit_rate_error <= 5.0
        }

        errors = [comp['error_pct'] for comp in comparison.values() if 'error_pct' in comp]
        avg_error = np.mean(errors) if errors else 0
        passed = avg_error <= 5.0

        return {
            'passed': passed, 'avg_error_pct': avg_error,
            'comparison': comparison, 'trace_stats': self.trace_stats
        }


# ============================================================================ 
# USER SIMULATION AND WORKLOAD GENERATION
# Creates a realistic mix of user behaviors and request patterns.
# ============================================================================ 

class UserSimulator:
    """Generates realistic user workloads based on pre-defined templates."""

    # Templates for different user personas (chatbot, coding, document analysis).
    USER_TEMPLATES = {
        'chatbot': {
            'context_range': (256, 1024), 'generation_range': (50, 150), 'think_time_range': (0.1, 0.5),
        },
        'coding': {
            'context_range': (1024, 4096), 'generation_range': (100, 500), 'think_time_range': (0.2, 1.0),
        },
        'document': {
            'context_range': (2048, 8192), 'generation_range': (200, 800), 'think_time_range': (0.3, 1.5),
        },
    }

    @classmethod
    def generate_user(cls, user_id: str, user_type: str = 'chatbot', priority: int = 1,
                      qos_level: QoSLevel = QoSLevel.BATCH) -> UserProfile:
        """Generates a single user profile based on a template."""
        template = cls.USER_TEMPLATES.get(user_type, cls.USER_TEMPLATES['chatbot'])
        return UserProfile(
            user_id=user_id,
            context_length=random.randint(*template['context_range']),
            generation_length=random.randint(*template['generation_range']),
            think_time=random.uniform(*template['think_time_range']),
            priority=priority,
            qos_level=qos_level
        )

    @classmethod
    def generate_mixed_users(cls, num_users: int) -> List[UserProfile]:
        """Generates a list of users with a realistic distribution of types and QoS levels."""
        users = []
        for i in range(num_users):
            user_type = random.choice(['chatbot', 'coding', 'document'])

            # Simulate a realistic QoS distribution.
            # 15% Interactive, 35% Responsive, 50% Batch.
            rand = random.random()
            if rand < 0.15:
                qos_level, priority = QoSLevel.INTERACTIVE, 3
            elif rand < 0.50:
                qos_level, priority = QoSLevel.RESPONSIVE, 2
            else:
                qos_level, priority = QoSLevel.BATCH, 1

            users.append(cls.generate_user(f"user_{i:04d}", user_type, priority, qos_level))
        return users


# ============================================================================ 
# INTEGRATED BENCHMARK ORCHESTRATOR
# This class wires all the components together and runs the main benchmark loop.
# ============================================================================ 

class IntegratedBenchmark:
    """The main orchestrator for the entire benchmark."""

    def __init__(self,
                 model_config: ModelConfig,
                 num_users: int,
                 gpu_memory_gb: float,
                 cpu_memory_gb: float,
                 duration_seconds: int,
                 cache_dir: str = None,
                 enable_autoscaling: bool = False,
                 autoscaler_mode: str = 'qos',
                 target_saturation: float = 0.8,
                 enable_multi_turn: bool = True,
                 enable_prefix_caching: bool = True,
                 enable_rag: bool = False,
                 rag_num_docs: int = 10,
                 validation_trace: Optional[str] = None,
                 generation_mode: GenerationMode = GenerationMode.NONE,
                 performance_profile: str = 'latency',
                 use_burst_trace: bool = False,
                 burst_trace_path: Optional[str] = None,
                 seed: Optional[int] = None):

        self.model_config = model_config
        self.num_users = num_users
        self.initial_users = num_users
        self.duration = duration_seconds
        self.enable_autoscaling = enable_autoscaling
        self.enable_multi_turn = enable_multi_turn
        self.generation_mode = generation_mode
        self.ms_per_token = GENERATION_TIMING[generation_mode] * 1000
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_rag = enable_rag
        self.rag_num_docs = rag_num_docs
        self.performance_profile = performance_profile
        self.use_burst_trace = use_burst_trace
        self.burst_trace_path = burst_trace_path
        self.seed = seed
        self.burst_requests: List[Tuple[int, int]] = []
        if self.use_burst_trace:
            self._load_burst_trace()

        # Initialize components
        self.cache = MultiTierCache(
            model_config=model_config,
            gpu_memory_gb=gpu_memory_gb,
            cpu_memory_gb=cpu_memory_gb,
            cache_dir=cache_dir,
            performance_profile=performance_profile,
            seed=seed
        )
        self.conversation_manager = ConversationManager()
        self.prefix_cache_manager = PrefixCacheManager(self.cache) if enable_prefix_caching else None
        self.rag_manager = RAGDocumentManager(self.cache) if enable_rag else None
        self.qos_monitor = QoSMonitor()
        self.storage_monitor = StorageMonitor(self) if enable_autoscaling else None
        self.autoscaler = WorkloadAutoscaler(
            mode=autoscaler_mode,
            initial_users=self.num_users,
            target_saturation=target_saturation
        ) if enable_autoscaling else None
        self.scale_interval = self.autoscaler.scale_interval if self.autoscaler else 1.0
        self.validator = ValidationEngine(validation_trace) if validation_trace else None

        self.request_queue = queue.PriorityQueue()
        self.request_counter = 0
        self.counter_lock = threading.Lock()

        self.active_users = []
        self.user_generators = {}
        self.user_conversations: Dict[str, str] = {}
        self.user_conversations_lock = threading.Lock()

        # Dictionary to store all results.
        self.results = {
            'requests_completed': 0, 'total_tokens_generated': 0,
            'total_storage_io_latency': 0.0, 'total_generation_latency': 0.0,
            'end_to_end_latencies': [], 'storage_latencies': [], 'generation_latencies': [],
            'throughput_timeline': [], 'prefill_latencies': [], 'decode_latencies': [],
            'multi_turn_cache_hits': 0, 'multi_turn_cache_misses': 0,
            'seed': self.seed,
        }
        self.results_lock = threading.Lock()
        self.rag_ingest_done = threading.Event() if self.enable_rag else None

    def _ingest_rag_documents(self, num_docs: int, stop_event: Optional[threading.Event] = None):
        """Ingests RAG documents for the workload."""
        print(f"Ingesting {num_docs} RAG documents...")
        for i in range(num_docs):
            if stop_event and stop_event.is_set():
                break
            # Scale document size based on model footprint so ingestion doesn't monopolize memory.
            if self.model_config.hidden_dim >= 8192 or self.model_config.num_layers >= 64:
                token_range = (1024, 4096)
            else:
                token_range = (4000, 12000)

            doc_tokens = random.randint(*token_range)
            self.rag_manager.ingest_document(f"doc_{i:04d}", doc_tokens, self.model_config)

        if self.rag_ingest_done:
            self.rag_ingest_done.set()

    def _load_burst_trace(self):
        """Loads requests from the BurstGPT CSV trace file."""
        if not self.burst_trace_path:
            print("Error: --use-burst-trace flag requires --burst-trace-path to be set.")
            sys.exit(1)
        try:
            with open(self.burst_trace_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        context_tokens = int(row['Request tokens'])
                        generate_tokens = int(row['Response tokens'])
                        self.burst_requests.append((context_tokens, generate_tokens))
                    except (ValueError, KeyError):
                        continue
            print(f"Loaded {len(self.burst_requests)} requests from BurstGPT trace.")
        except FileNotFoundError:
            print(f"Error: Trace file not found at {self.burst_trace_path}")
            sys.exit(1)
        except Exception as e:
            print(f"Error reading trace file: {e}")
            sys.exit(1)

    def _generate_requests_from_trace(self, stop_event: threading.Event):
        """Generates InferenceRequest objects from the loaded trace."""
        request_index = 0
        while not stop_event.is_set():
            if not self.burst_requests:
                print("Warning: BurstGPT trace is empty. No requests to generate.")
                time.sleep(1)
                continue

            if request_index >= len(self.burst_requests):
                request_index = 0 # Loop

            context_tokens, generate_tokens = self.burst_requests[request_index]

            with self.counter_lock:
                req_id = self.request_counter
                self.request_counter += 1

            rand = random.random()
            if rand < 0.15:
                qos_level, priority = QoSLevel.INTERACTIVE, 3
            elif rand < 0.50:
                qos_level, priority = QoSLevel.RESPONSIVE, 2
            else:
                qos_level, priority = QoSLevel.BATCH, 1
            
            user_id = f"trace_user_{request_index % 1000}"

            # Determine inference phase for trace-driven requests.
            # CRITICAL FIX: Using the same 10000-token threshold as synthetic workloads
            # to ensure consistent behavior and comprehensive storage I/O testing.
            # See the detailed explanation in generate_requests() for why this threshold matters.
            request = InferenceRequest(
                user_id=user_id,
                request_id=f"{user_id}_req_{req_id:04d}",
                timestamp=datetime.now(),
                context_tokens=context_tokens,
                generate_tokens=generate_tokens,
                priority=priority,
                phase=InferencePhase.PREFILL if context_tokens >= 10000 else InferencePhase.PREFILL_DECODE,
                qos_level=qos_level,
                cache_key=f"{user_id}_req_{req_id:04d}"
            )

            priority_tuple = (-QOS_PROFILES[request.qos_level].priority, time.time())
            self.request_queue.put((priority_tuple, request))
            
            request_index += 1
            time.sleep(0.01) # Simulate request arrival rate

    def generate_requests(self, users: List[UserProfile], stop_event: threading.Event):
        """Generate requests concurrently for each simulated user."""

        # Kick off RAG ingestion so document threads can run in parallel with user traffic.
        if self.enable_rag and self.rag_manager and self.rag_ingest_done:
            threading.Thread(
                target=self._ingest_rag_documents,
                args=(self.rag_num_docs, stop_event),
                daemon=True
            ).start()

        def enqueue_request(request: InferenceRequest):
            priority_tuple = (-QOS_PROFILES[request.qos_level].priority, time.time())
            self.request_queue.put((priority_tuple, request))

        def user_worker(user: UserProfile):
            """Simulates an individual user generating traffic."""
            local_conv_id = None

            while not stop_event.is_set():
                # Randomize think time slightly to avoid global synchronization.
                time.sleep(user.think_time * random.uniform(0.8, 1.2))
                if stop_event.is_set():
                    break

                # Handle conversation lifecycle when multi-turn is enabled.
                if self.enable_multi_turn and self.conversation_manager:
                    if local_conv_id and random.random() >= 0.8:
                        with self.user_conversations_lock:
                            self.user_conversations.pop(user.user_id, None)
                        local_conv_id = None

                    if local_conv_id is None:
                        local_conv_id = self.conversation_manager.start_conversation(user.user_id)
                        with self.user_conversations_lock:
                            self.user_conversations[user.user_id] = local_conv_id
                else:
                    local_conv_id = None

                new_context = random.randint(max(1, user.context_length // 4), user.context_length)
                new_gen = random.randint(max(1, user.generation_length // 4), user.generation_length)

                with self.counter_lock:
                    req_id = self.request_counter
                    self.request_counter += 1

                if self.enable_multi_turn and self.conversation_manager and local_conv_id:
                    turn_number, cache_key = self.conversation_manager.add_turn(local_conv_id, new_context, new_gen)
                else:
                    turn_number = 1
                    cache_key = f"{user.user_id}_req_{req_id:06d}"

                phase = InferencePhase.PREFILL if new_context >= 10000 else InferencePhase.PREFILL_DECODE

                request = InferenceRequest(
                    user_id=user.user_id,
                    request_id=f"req_{user.user_id}_{req_id:06d}",
                    timestamp=datetime.now(),
                    context_tokens=new_context,
                    generate_tokens=new_gen,
                    priority=user.priority,
                    phase=phase,
                    qos_level=user.qos_level,
                    cache_key=cache_key,
                    conversation_id=local_conv_id,
                    turn_number=turn_number
                )

                enqueue_request(request)

                # Occasionally inject RAG queries on behalf of this user.
                if (self.enable_rag and self.rag_manager and self.rag_ingest_done and
                        self.rag_ingest_done.is_set() and self.rag_manager.documents and
                        random.random() < 0.1):
                    doc_id = random.choice(list(self.rag_manager.documents.keys()))
                    retrieved_chunks = self.rag_manager.retrieve_chunks(doc_id)
                    rag_context_tokens = sum(chunk.token_count for chunk in retrieved_chunks)

                    with self.counter_lock:
                        rag_req_id = self.request_counter
                        self.request_counter += 1

                    rag_request = InferenceRequest(
                        user_id=user.user_id,
                        request_id=f"rag_{user.user_id}_{rag_req_id:06d}",
                        timestamp=datetime.now(),
                        context_tokens=rag_context_tokens,
                        generate_tokens=random.randint(50, 200),
                        priority=user.priority,
                        phase=InferencePhase.DECODE,
                        qos_level=user.qos_level,
                        cache_key=f"rag_{doc_id}"
                    )
                    enqueue_request(rag_request)

        # Launch a worker thread per user to maintain high request concurrency.
        for user in users:
            threading.Thread(target=user_worker, args=(user,), daemon=True).start()

        self.active_users = users

        # Keep this generator alive until the benchmark signals shutdown.
        stop_event.wait()

    def process_requests(self, stop_event: threading.Event):
        """The main worker loop that processes requests from the queue."""
        while not stop_event.is_set():
            try:
                priority_tuple, request = self.request_queue.get(timeout=0.5)
            except queue.Empty:
                continue # If the queue is empty, loop again.

            request.start_time = time.perf_counter()
            storage_latency = 0.0
            cache_type = 'user'

            # --- REQUEST LIFECYCLE --- #

            # 1. Check for a prefix cache hit.
            if self.prefix_cache_manager:
                prefix_entry, remaining_tokens = self.prefix_cache_manager.check_prefix_cache(request, self.model_config)
                if prefix_entry:
                    cache_type = 'system' if prefix_entry.prefix_type == PrefixType.SYSTEM_PROMPT else 'common'
                    _, read_lat = self.cache.access_cache(prefix_entry.kv_cache_key, request.phase, cache_type)
                    storage_latency += read_lat
                    request.context_tokens = remaining_tokens

            # 2. For multi-turn conversations, access the cache from the previous turn.
            if self.conversation_manager and request.turn_number > 1:
                prev_turn_key = f"{request.conversation_id}_turn_{request.turn_number - 1}"
                location, read_latency = self.cache.access_cache(prev_turn_key, InferencePhase.DECODE, 'multi_turn')
                if location is not None:
                    storage_latency += read_latency
                    with self.results_lock: self.results['multi_turn_cache_hits'] += 1
                else:
                    with self.results_lock: self.results['multi_turn_cache_misses'] += 1

            # 3. Perform the main PREFILL operation (a cache WRITE).
            if request.phase == InferencePhase.PREFILL or request.phase == InferencePhase.PREFILL_DECODE:
                success, location, write_latency = self.cache.allocate_cache(
                    request.cache_key, request.context_tokens, InferencePhase.PREFILL
                )
                storage_latency += write_latency
                with self.results_lock: self.results['prefill_latencies'].append(write_latency)

            # 4. Simulate a RAG operation by reading random chunk caches.
            if self.rag_manager and random.random() < 0.1: # 10% of requests are RAG queries
                doc_id = random.choice(list(self.rag_manager.documents.keys()))
                chunks = self.rag_manager.retrieve_chunks(doc_id)
                for chunk in chunks: # Read the KV cache for each retrieved chunk.
                    _, read_lat = self.cache.access_cache(chunk.kv_cache_key, InferencePhase.DECODE)
                    storage_latency += read_lat

            # 5. Perform the DECODE operation (a cache READ).
            if request.phase == InferencePhase.DECODE or request.phase == InferencePhase.PREFILL_DECODE:
                location, read_latency = self.cache.access_cache(request.cache_key, InferencePhase.DECODE, cache_type)

                if location is None: # This would be a cache miss.
                    _, _, write_latency = self.cache.allocate_cache(
                        request.cache_key,
                        request.context_tokens,
                        InferencePhase.PREFILL
                    )
                    storage_latency += write_latency
                else:
                    # Simulate realistic decode I/O: reads are batched, not per-token.
                    decode_batch_size = 32
                    num_batched_reads = max(1, (request.generate_tokens + decode_batch_size - 1) // decode_batch_size)
                    for _ in range(num_batched_reads):
                        _, batch_read_latency = self.cache.access_cache(request.cache_key, InferencePhase.DECODE, cache_type)
                        storage_latency += batch_read_latency

                with self.results_lock: self.results['decode_latencies'].append(read_latency)

            # 6. Simulate token generation time if not in pure storage mode.
            generation_latency = request.generate_tokens * GENERATION_TIMING[self.generation_mode]
            if generation_latency > 0: time.sleep(generation_latency)

            request.complete_time = time.perf_counter()

            # 7. Record all results for this request.
            with self.results_lock:
                self.results['requests_completed'] += 1
                self.results['total_tokens_generated'] += request.generate_tokens
                self.results['total_storage_io_latency'] += storage_latency
                self.results['total_generation_latency'] += generation_latency
                self.results['end_to_end_latencies'].append(request.total_latency_ms / 1000)
                self.results['storage_latencies'].append(storage_latency)
                self.results['generation_latencies'].append(generation_latency)

            self.qos_monitor.record_request(request)

    def monitor_stats(self, stop_event: threading.Event):
        """Periodically collects and logs stats, and triggers autoscaling."""
        start_time = time.time()
        last_log_time = start_time

        while not stop_event.is_set():
            time.sleep(self.scale_interval)
            now = time.time()

            elapsed = now - start_time
            if elapsed > self.duration:
                break

            # Track throughput timeline for reporting
            with self.results_lock:
                total_tokens = self.results['total_tokens_generated']
            throughput = total_tokens / max(elapsed, 1e-6)
            with self.results_lock:
                self.results['throughput_timeline'].append({
                    'timestamp': elapsed,
                    'throughput_tokens_per_sec': throughput
                })

            if self.enable_autoscaling and self.storage_monitor and self.autoscaler:
                metrics = self.storage_monitor.collect_metrics(self.cache, self.request_queue.qsize())
                saturation_level = self.storage_monitor.get_saturation_level()
                if metrics:
                    metrics.saturation_level = saturation_level

                action, target_users = self.autoscaler.calculate_scale_action(
                    metrics if metrics else None,
                    throughput,
                    saturation_level
                )

                if action in ('scale_up', 'scale_down') and target_users != self.num_users:
                    self.num_users = max(1, min(target_users, 500))
                    self.autoscaler.current_users = self.num_users
                    log_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'mode': self.autoscaler.mode,
                        'action': action,
                        'users': self.num_users,
                        'saturation_level': saturation_level,
                        'read_latency_p95_ms': metrics.read_latency_p95_ms if metrics else None,
                        'write_latency_p95_ms': metrics.write_latency_p95_ms if metrics else None,
                        'throughput_tokens_per_sec': throughput
                    }
                    self.autoscaler.scaling_history.append(log_entry)
                    print(f"Autoscaler {action} -> {self.num_users} users (saturation: {saturation_level:.2f})")
                elif action == 'stop':
                    print("Autoscaler requested stop after reaching capacity peak.")
                    stop_event.set()
                    log_entry = {
                        'timestamp': datetime.now().isoformat(),
                        'mode': self.autoscaler.mode,
                        'action': 'stop',
                        'users': self.num_users,
                        'saturation_level': saturation_level,
                        'peak_throughput_tokens_per_sec': self.autoscaler.peak_throughput
                    }
                    self.autoscaler.scaling_history.append(log_entry)
                else:
                    # Keep autoscaler internal state aligned with the active user count.
                    self.autoscaler.current_users = self.num_users

            # Log stats periodically
            if now - last_log_time >= 10:
                self._calculate_stats()
                queue_depth = self.request_queue.qsize()
                print(f"Time: {int(elapsed)}s, Users: {self.num_users}, Queue: {queue_depth}, "
                      f"Throughput: {throughput:.2f} tok/s")
                last_log_time = now

    def run(self) -> Dict:
        """The main entry point to start the benchmark execution."""
        print(f"\nIntegrated Multi-User KV Cache Benchmark - MLPerf Edition")
        print(f"Model: {self.model_config.name}")
        print(f"Users: {self.num_users}")
        print(f"Duration: {self.duration}s")
        if self.seed is not None:
            print(f"Seed: {self.seed}")
        print(f"Generation Mode: {self.generation_mode.value} ({self.ms_per_token:.1f}ms/token)")
        print(f"Features:")
        print(f"  - Phase-Aware Processing: Enabled")
        print(f"  - Multi-turn Conversations: {'Enabled' if self.enable_multi_turn else 'Disabled'}")
        print(f"  - Prefix Caching: {'Enabled' if self.enable_prefix_caching else 'Disabled'}")
        print(f"  - RAG Workload: {'Enabled' if self.enable_rag else 'Disabled'}")
        print(f"  - Autoscaling: {'Enabled' if self.enable_autoscaling else 'Disabled'}")
        if self.enable_autoscaling:
            print(f"    - Mode: {self.autoscaler.mode}")
        print(f"  - QoS Support: Enabled (Interactive/Responsive/Batch)")
        print(f"  - Trace-Driven (BurstGPT): {'Enabled' if self.use_burst_trace else 'Disabled'}")
        print("=" * 80)

        users = []
        if not self.use_burst_trace:
            users = UserSimulator.generate_mixed_users(self.num_users)
            context_lengths = [u.context_length for u in users]
            print(f"\nUser Context Length Distribution:")
            print(f"  Min: {min(context_lengths)} tokens ({min(context_lengths) * self.model_config.kv_cache_size_per_token / 1024**2:.2f} MB)")
            print(f"  Max: {max(context_lengths)} tokens ({max(context_lengths) * self.model_config.kv_cache_size_per_token / 1024**2:.2f} MB)")
            print(f"  Mean: {np.mean(context_lengths):.0f} tokens ({np.mean(context_lengths) * self.model_config.kv_cache_size_per_token / 1024**2:.2f} MB)")

            qos_dist = {level: sum(1 for u in users if u.qos_level == level) for level in QoSLevel}
            print(f"\nQoS Distribution:")
            for level, count in qos_dist.items():
                print(f"  {level.value}: {count} users")

        print(f"\nStarting benchmark...")
        print("-" * 80)

        stop_event = threading.Event()

        threads = []
        if self.use_burst_trace:
            gen_thread = threading.Thread(target=self._generate_requests_from_trace, args=(stop_event,), daemon=True)
        else:
            gen_thread = threading.Thread(target=self.generate_requests, args=(users, stop_event), daemon=True)
        
        threads.append(gen_thread)
        gen_thread.start()

        num_workers = min(self.num_users, 500)
        for _ in range(num_workers):
            proc_thread = threading.Thread(target=self.process_requests, args=(stop_event,), daemon=True)
            threads.append(proc_thread)
            proc_thread.start()

        # Only start the monitor thread if autoscaling is enabled.
        if self.enable_autoscaling:
            mon_thread = threading.Thread(target=self.monitor_stats, args=(stop_event,), daemon=True)
            threads.append(mon_thread)
            mon_thread.start()

        # Wait for either the configured duration or an earlier stop signal from the monitor.
        stop_event.wait(timeout=self.duration)

        stop_event.set()
        for thread in threads:
            thread.join(timeout=2.0)

        self._calculate_stats()

        if self.validator:
            self.results['validation'] = self.validator.validate_benchmark(self.results)

        return self.results

    def _calculate_stats(self):
        """Calculate final statistics with all feature breakdowns"""
        if not self.results['end_to_end_latencies']:
            print("\nNo requests completed during benchmark!")
            return

        e2e = np.array(self.results['end_to_end_latencies'])
        storage = np.array(self.results['storage_latencies'])
        generation = np.array(self.results['generation_latencies'])

        cache_stats = self.cache.get_stats(self.duration)
        qos_metrics = self.qos_monitor.get_all_qos_metrics()
        prefix_stats = self.prefix_cache_manager.stats if self.prefix_cache_manager else {}
        autoscaling_stats = self.autoscaler.scaling_history if self.autoscaler else []

        autoscaling_summary = None
        if self.autoscaler:
            autoscaling_summary = {
                'initial_users': getattr(self, 'initial_users', self.num_users),
                'final_users': self.autoscaler.current_users,
                'total_scale_events': len(autoscaling_stats)
            }
            if self.autoscaler.mode == 'capacity':
                autoscaling_summary.update({
                    'peak_user_count': self.autoscaler.peak_user_count,
                    'peak_throughput_tokens_per_sec': self.autoscaler.peak_throughput
                })

        summary = {
            'total_requests': self.results['requests_completed'],
            'total_tokens': self.results['total_tokens_generated'],
            'avg_throughput_tokens_per_sec': self.results['total_tokens_generated'] / self.duration,
            'requests_per_second': self.results['requests_completed'] / self.duration,
            'end_to_end_latency_ms': {
                'mean': np.mean(e2e) * 1000,
                'p50': np.percentile(e2e, 50) * 1000,
                'p95': np.percentile(e2e, 95) * 1000,
                'p99': np.percentile(e2e, 99) * 1000,
            },
            'storage_io_latency_ms': {
                'mean': np.mean(storage) * 1000,
                'p50': np.percentile(storage, 50) * 1000,
                'p95': np.percentile(storage, 95) * 1000,
                'p99': np.percentile(storage, 99) * 1000,
            },
            'generation_latency_ms': {
                'mean': np.mean(generation) * 1000,
                'p50': np.percentile(generation, 50) * 1000,
                'p95': np.percentile(generation, 95) * 1000,
                'p99': np.percentile(generation, 99) * 1000,
            },
            'cache_stats': cache_stats,
            'qos_metrics': qos_metrics,
            'prefix_cache_stats': prefix_stats,
            'autoscaling_stats': autoscaling_stats,
            'autoscaling_summary': autoscaling_summary,
            'multi_turn_stats': {
                'cache_hits': self.results['multi_turn_cache_hits'],
                'cache_misses': self.results['multi_turn_cache_misses'],
                'hit_rate': self.results['multi_turn_cache_hits'] /
                           max(self.results['multi_turn_cache_hits'] + self.results['multi_turn_cache_misses'], 1)
            }
        }
        self.results['summary'] = summary
        self._print_summary(summary)
    
    def _print_summary(self, summary: Dict):
        """
        Print a comprehensive benchmark results summary to console.
        Displays detailed performance metrics including storage I/O latency, throughput,
        cache statistics, tier-specific performance, and QoS metrics in a formatted
        report suitable for analysis and comparison.
        Args:
            summary (Dict): Benchmark results dictionary containing:
                - cache_stats: Storage performance and cache hit statistics
                - total_requests: Number of completed requests
                - total_tokens: Total tokens processed
                - avg_throughput_tokens_per_sec: Average token throughput
                - requests_per_second: Request rate
                - end_to_end_latency_ms: Complete request latency percentiles
                - storage_io_latency_ms: Storage-only latency percentiles  
                - generation_latency_ms: Token generation latency percentiles
                - qos_metrics: Quality of service metrics by tier
                - prefix_cache_stats: Prefix caching performance (optional)
                - multi_turn_stats: Multi-turn conversation metrics (optional)
                - autoscaling_stats: Autoscaling events (optional)
        The report includes:
            - Storage performance assessment with pass/fail criteria
            - Overall throughput and latency metrics
            - Cache hit rates and I/O statistics
            - Memory tier distribution (GPU/CPU/NVMe)
            - Phase-specific metrics (prefill/decode)
            - QoS compliance by service tier
            - Validation results if available
        Note:
            The symbols âœ" and âœ— are intended to be checkmark (✓) and cross (✗) 
            characters for pass/fail indicators but may display incorrectly due to 
            encoding issues.
        """
        """Print comprehensive results summary"""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS - MLPerf KV Cache Storage Benchmark")
        print(f"Generation Mode: {self.generation_mode.value} ({self.ms_per_token:.1f}ms/token)")
        print("=" * 80)

        cache_stats = summary['cache_stats']
        if 'storage_health' in cache_stats:
            storage_health = cache_stats['storage_health']
            status = storage_health['overall_status']
            status_symbol = 'âœ“' if status == 'PASS' else 'âœ—'
            print(f"\n### STORAGE PERFORMANCE ASSESSMENT: {status} {status_symbol} ###")
            print(f"  Criteria Passed: {storage_health['passed_count']}/{storage_health['total_count']}")
            for criterion in storage_health['criteria']:
                symbol = 'âœ“' if criterion['passed'] else 'âœ—'
                unit = criterion.get('unit', '')
                if unit == 'ratio':
                    print(f"  {symbol} {criterion['name']}: {criterion['actual']:.1%} (target: {criterion['target']:.1%})")
                    continue

                actual = criterion.get('actual')
                target = criterion.get('target')
                try:
                    # Attempt to format if it's a number
                    actual_str = f"{actual:.2f}"
                except (ValueError, TypeError):
                    # If it's already a string or can't be formatted, use it directly
                    actual_str = str(actual)

                try:
                    target_str = f"{target:.2f}"
                except (ValueError, TypeError):
                    target_str = str(target)

                unit_suffix = unit if unit else ''
                print(f"  {symbol} {criterion['name']}: {actual_str}{unit_suffix} (target: {target_str}{unit_suffix})")

        print(f"\n### OVERALL PERFORMANCE ###")
        print(f"Requests Completed: {summary['total_requests']}")
        print(f"Total Tokens Generated: {summary['total_tokens']}")
        print(f"Throughput: {summary['avg_throughput_tokens_per_sec']:.2f} tokens/sec")
        print(f"Requests/sec: {summary['requests_per_second']:.2f}")

        print(f"\n### END-TO-END LATENCY (Storage I/O + Token Generation) ###")
        print(f"  Mean: {summary['end_to_end_latency_ms']['mean']:.2f} ms")
        print(f"  P50:  {summary['end_to_end_latency_ms']['p50']:.2f} ms")
        print(f"  P95:  {summary['end_to_end_latency_ms']['p95']:.2f} ms")
        print(f"  P99:  {summary['end_to_end_latency_ms']['p99']:.2f} ms")

        print(f"\n### STORAGE I/O LATENCY (Primary Metric) ###")
        print(f"  Mean: {summary['storage_io_latency_ms']['mean']:.2f} ms")
        print(f"  P50:  {summary['storage_io_latency_ms']['p50']:.2f} ms")
        print(f"  P95:  {summary['storage_io_latency_ms']['p95']:.2f} ms")
        print(f"  P99:  {summary['storage_io_latency_ms']['p99']:.2f} ms")

        if self.generation_mode != GenerationMode.NONE:
            print(f"\n### TOKEN GENERATION LATENCY (Simulated @ {self.ms_per_token:.1f}ms/token) ###")
            print(f"  Mean: {summary['generation_latency_ms']['mean']:.2f} ms")
            print(f"  P50:  {summary['generation_latency_ms']['p50']:.2f} ms")
            print(f"  P95:  {summary['generation_latency_ms']['p95']:.2f} ms")

        print(f"\n### STORAGE PERFORMANCE ###")
        print(f"  Cache Hit Rate: {cache_stats['cache_hit_rate']*100:.1f}%")
        print(f"  Total Read: {cache_stats['total_read_gb']:.2f} GB")
        print(f"  Total Write: {cache_stats['total_write_gb']:.2f} GB")
        print(f"  Read/Write Ratio: {cache_stats['read_write_ratio']:.2f}")
        print(f"  Read IOPS: {cache_stats['read_iops'] / self.duration:.2f}")
        print(f"  Write IOPS: {cache_stats['write_iops'] / self.duration:.2f}")

        print(f"\n### CACHE TIER DISTRIBUTION ###")
        print(f"  GPU Entries: {cache_stats['gpu_entries']} ({cache_stats['gpu_memory_used_gb']:.2f} GB)")
        print(f"  CPU Entries: {cache_stats['cpu_entries']} ({cache_stats['cpu_memory_used_gb']:.2f} GB)")
        print(f"  NVMe Entries: {cache_stats['nvme_entries']}")

        print(f"\n### PHASE-SPECIFIC METRICS ###")
        print(f"  Prefill Writes: {cache_stats['prefill_writes']}")
        print(f"  Prefill Bytes Written: {cache_stats['prefill_bytes_written_gb']:.2f} GB")
        print(f"  Decode Reads: {cache_stats['decode_reads']}")
        print(f"  Decode Bytes Read: {cache_stats['decode_bytes_read_gb']:.2f} GB")

        print(f"\n### TIER-SPECIFIC LATENCIES ###")
        for tier in ['gpu', 'cpu', 'nvme']:
            for op in ['read', 'write']:
                p95_key = f'{tier}_{op}_p95_ms'
                if p95_key in cache_stats:
                    print(f"  {tier.upper()} {op.title()} P95: {cache_stats[p95_key]:.2f} ms")

        print(f"\n### CACHE TYPE BREAKDOWNS ###")
        print(f"  System Prompt Hits: {cache_stats['system_prompt_hits']}")
        print(f"  Common Phrase Hits: {cache_stats['common_phrase_hits']}")
        print(f"  User Cache Hits: {cache_stats['user_cache_hits']}")
        print(f"  Multi-turn Hits: {cache_stats['multi_turn_hits']}")

        if summary.get('prefix_cache_stats') and summary['prefix_cache_stats']['prefix_hits'] > 0:
            print(f"\n### PREFIX CACHING ###")
            prefix_stats = summary['prefix_cache_stats']
            print(f"  Prefix Hits: {prefix_stats['prefix_hits']}")
            print(f"  Prefix Misses: {prefix_stats['prefix_misses']}")
            print(f"  System Prompt Reuse: {prefix_stats['system_prompt_reuse']}")
            print(f"  Bytes Saved: {prefix_stats['bytes_saved'] / 1024**3:.2f} GB")

        if summary.get('multi_turn_stats') and summary['multi_turn_stats']['cache_hits'] > 0:
            print(f"\n### MULTI-TURN CONVERSATIONS ###")
            mt_stats = summary['multi_turn_stats']
            print(f"  Multi-turn Cache Hits: {mt_stats['cache_hits']}")
            print(f"  Multi-turn Cache Misses: {mt_stats['cache_misses']}")
            print(f"  Multi-turn Hit Rate: {mt_stats['hit_rate']*100:.1f}%")

        print(f"\n### QOS LATENCY METRICS (Informational - includes simulated generation) ###")
        qos_metrics = summary['qos_metrics']
        for qos_level, metrics in qos_metrics.items():
            if metrics.get('no_data'): continue
            print(f"\n  {qos_level.upper()}:")
            print(f"    Requests: {metrics['total_requests']}")
            print(f"    Latency P95: {metrics['latency_ms']['p95']:.2f} ms")
            print(f"    Latency P99: {metrics['latency_ms']['p99']:.2f} ms")
            if 'sla' in metrics:
                sla_met = 'âœ“' if metrics['sla']['met'] else 'âœ—'
                print(f"    SLA Met: {sla_met} (compliance: {metrics['sla']['compliance']:.1%})")

        if summary.get('autoscaling_stats'):
            auto_stats = summary['autoscaling_stats']
            if auto_stats:
                print(f"\n### AUTOSCALING ({self.autoscaler.mode} mode) ###")
                print(f"  Scaling Events: {len(auto_stats)}")
                print(f"  Final User Count: {self.autoscaler.current_users}")
                if self.autoscaler.mode == 'capacity':
                    print(f"  Peak Capacity Found: {self.autoscaler.peak_throughput:.2f} tok/s at {self.autoscaler.peak_user_count} users")

        if 'validation' in self.results:
            print(f"\n### VALIDATION ###")
            validation = self.results['validation']
            print(f"  Validation: {'PASSED âœ“' if validation['passed'] else 'FAILED âœ—'}")
            print(f"  Average Error: {validation['avg_error_pct']:.2f}%")

        print("\n" + "=" * 80)
        print("NOTES:")
        if self.generation_mode == GenerationMode.NONE:
            print("  - Pure storage I/O benchmark (no generation simulation)")
        else:
            print("  - End-to-end latency includes simulated GPU inference")
        print("=" * 80)


def main():
    """Main entry point for running the benchmark from the command line."""
    parser = argparse.ArgumentParser(description="Integrated Multi-User KV Cache Benchmark")
    parser.add_argument('--model', type=str, default='llama3.1-8b', choices=MODEL_CONFIGS.keys(),
                        help='The model configuration to use.')
    parser.add_argument('--num-users', type=int, default=100,
                        help='The number of concurrent users to simulate.')
    parser.add_argument('--duration', type=int, default=60,
                        help='The duration of the benchmark in seconds.')
    parser.add_argument('--gpu-mem-gb', type=float, default=16,
                        help='The amount of GPU memory (VRAM) to allocate for the cache in GB.')
    parser.add_argument('--cpu-mem-gb', type=float, default=32,
                        help='The amount of CPU memory (RAM) to allocate for the cache in GB.')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='The directory to use for the NVMe cache tier. Defaults to a temporary directory.')
    parser.add_argument('--generation-mode', type=str, default='realistic', choices=[g.value for g in GenerationMode],
                        help='The token generation speed simulation mode.')
    parser.add_argument('--performance-profile', type=str, default='latency', choices=['latency', 'throughput'],
                        help='The performance profile to use for pass/fail criteria (latency or throughput).')
    parser.add_argument('--disable-multi-turn', action='store_true',
                        help='Disable multi-turn conversation caching.')
    parser.add_argument('--disable-prefix-caching', action='store_true',
                        help='Disable prefix caching.')
    parser.add_argument('--enable-rag', action='store_true',
                        help='Enable the RAG workload simulation.')
    parser.add_argument('--rag-num-docs', type=int, default=10, help='Number of RAG documents to ingest')
    parser.add_argument('--enable-autoscaling', action='store_true',
                        help='Enable workload autoscaling.')
    parser.add_argument('--autoscaler-mode', type=str, default='qos', choices=['qos', 'capacity'],
                        help='The autoscaling strategy: "qos" (latency-based) or "capacity" (throughput-based).')
    parser.add_argument('--target-saturation', type=float, default=0.8, help='Target storage saturation for autoscaling (0.0-1.0)')
    parser.add_argument('--use-burst-trace', action='store_true',
                        help='Use BurstGPT trace for workload generation.')
    parser.add_argument('--burst-trace-path', type=str, default='BurstGPT/data/BurstGPT_1.csv',
                        help='Path to the BurstGPT trace file.')
    parser.add_argument('--validation-trace', type=str, default=None,
                        help='Path to a real-world trace file for validation.')
    parser.add_argument('--output', type=str, default=f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", help='Output file for results')
    parser.add_argument('--seed', type=int, default=None,
                        help='Seed for random number generators to ensure reproducibility.')

    args = parser.parse_args()

    if args.seed is not None:
        print(f"Using random seed: {args.seed}")
        random.seed(args.seed)
        np.random.seed(args.seed)
        if TORCH_AVAILABLE:
            torch.manual_seed(args.seed)
        if CUPY_AVAILABLE:
            cp.random.seed(args.seed)

    model_config = MODEL_CONFIGS[args.model]
    gen_mode = GenerationMode(args.generation_mode)

    benchmark = IntegratedBenchmark(
        model_config=model_config,
        num_users=args.num_users,
        gpu_memory_gb=args.gpu_mem_gb,
        cpu_memory_gb=args.cpu_mem_gb,
        duration_seconds=args.duration,
        cache_dir=args.cache_dir,
        enable_autoscaling=args.enable_autoscaling,
        autoscaler_mode=args.autoscaler_mode,
        target_saturation=args.target_saturation,
        enable_multi_turn=not args.disable_multi_turn,
        enable_prefix_caching=not args.disable_prefix_caching,
        enable_rag=args.enable_rag,
        rag_num_docs=args.rag_num_docs,
        validation_trace=args.validation_trace,
        generation_mode=gen_mode,
        performance_profile=args.performance_profile,
        use_burst_trace=args.use_burst_trace,
        burst_trace_path=args.burst_trace_path,
        seed=args.seed
    )

    results = benchmark.run()

    # Save results to a JSON file
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, datetime):
            return obj.isoformat()
        if is_dataclass(obj):
            return asdict(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=4, default=convert_numpy)

    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    main()