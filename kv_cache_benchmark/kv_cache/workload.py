"""
Workload generation and validation for KV Cache Benchmark.

Contains ValidationEngine, UserSimulator, ShareGPTDatasetLoader,
and RealTraceEntry for trace-driven validation.
"""

import os
import json
import random
import logging
import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from kv_cache._compat import TIKTOKEN_AVAILABLE
from kv_cache.config import cfg
from kv_cache.models import (
    QoSLevel, UserProfile, InferenceRequest,
)

if TIKTOKEN_AVAILABLE:
    import tiktoken

logger = logging.getLogger(__name__)


# ============================================================================
# TRACE-DRIVEN VALIDATION
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
            return {
                'total_requests': 1000, 'duration_seconds': 100, 'cache_hit_rate': 0.65,
                'read_write_ratio': 10.0, 'context_tokens_mean': 1024, 'generation_tokens_mean': 200,
            }

        with open(self.trace_path, 'r') as f:
            data = json.load(f)
            entries = [RealTraceEntry(**entry) for entry in data]

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
# INPUT VALIDATION
# ============================================================================

# Validation constants with documented rationale
MAX_USERS = 100000
MAX_DURATION_SECONDS = 86400
MAX_GPU_MEMORY_GB = 65536   # supports up to 512 × 128 GB HBM per TP group (num_gpus × per-card)
MAX_CPU_MEMORY_GB = 131072  # supports up to 128 TB DRAM per node

FORBIDDEN_CACHE_PREFIXES = frozenset([
    '/etc', '/bin', '/sbin', '/usr/bin', '/usr/sbin',
    '/boot', '/sys', '/proc', '/dev', '/root'
])


def validate_args(args: argparse.Namespace) -> argparse.Namespace:
    """
    Validate command-line arguments to catch invalid values early.

    Args:
        args: Parsed argparse namespace

    Returns:
        The validated args namespace

    Raises:
        ValueError: If any validation check fails
    """
    errors = []

    if args.num_users <= 0:
        errors.append(f"--num-users must be positive, got {args.num_users}")
    if args.num_users > MAX_USERS:
        errors.append(f"--num-users exceeds limit ({MAX_USERS}), got {args.num_users}")

    if args.duration <= 0:
        errors.append(f"--duration must be positive, got {args.duration}")
    if args.duration > MAX_DURATION_SECONDS:
        errors.append(f"--duration exceeds 24 hours ({MAX_DURATION_SECONDS}s), got {args.duration}")

    if args.gpu_mem_gb < 0:
        errors.append(f"--gpu-mem-gb cannot be negative, got {args.gpu_mem_gb}")
    if args.gpu_mem_gb > MAX_GPU_MEMORY_GB:
        errors.append(f"--gpu-mem-gb exceeds limit ({MAX_GPU_MEMORY_GB}GB), got {args.gpu_mem_gb}")

    if args.cpu_mem_gb < 0:
        errors.append(f"--cpu-mem-gb cannot be negative, got {args.cpu_mem_gb}")
    if args.cpu_mem_gb > MAX_CPU_MEMORY_GB:
        errors.append(f"--cpu-mem-gb exceeds limit ({MAX_CPU_MEMORY_GB}GB), got {args.cpu_mem_gb}")

    if args.rag_num_docs < 0:
        errors.append(f"--rag-num-docs cannot be negative, got {args.rag_num_docs}")

    if args.max_conversations <= 0:
        errors.append(f"--max-conversations must be positive, got {args.max_conversations}")

    if args.max_concurrent_allocs < 0:
        errors.append(f"--max-concurrent-allocs cannot be negative, got {args.max_concurrent_allocs}")

    if args.request_rate < 0:
        errors.append(f"--request-rate cannot be negative, got {args.request_rate}")

    if args.max_requests < 0:
        errors.append(f"--max-requests cannot be negative, got {args.max_requests}")

    if args.storage_capacity_gb < 0:
        errors.append(f"--storage-capacity-gb cannot be negative, got {args.storage_capacity_gb}")

    if args.precondition_size_gb < 0:
        errors.append(f"--precondition-size-gb cannot be negative, got {args.precondition_size_gb}")

    if args.precondition_threads < 0:
        errors.append(f"--precondition-threads cannot be negative, got {args.precondition_threads}")

    if args.trace_speedup < 0:
        errors.append(f"--trace-speedup cannot be negative, got {args.trace_speedup}")

    if args.replay_cycles < 0:
        errors.append(f"--replay-cycles cannot be negative, got {args.replay_cycles}")

    if not (0.0 <= args.target_saturation <= 1.0):
        errors.append(f"--target-saturation must be between 0.0 and 1.0, got {args.target_saturation}")

    if args.num_gpus < 1:
        errors.append(f"--num-gpus must be >= 1, got {args.num_gpus}")

    if args.tensor_parallel < 1:
        errors.append(f"--tensor-parallel must be >= 1, got {args.tensor_parallel}")
    elif args.tensor_parallel > args.num_gpus:
        errors.append(
            f"--tensor-parallel ({args.tensor_parallel}) cannot exceed --num-gpus ({args.num_gpus})"
        )
    elif args.tensor_parallel > 1 and (args.tensor_parallel & (args.tensor_parallel - 1)) != 0:
        logger.warning(
            f"--tensor-parallel={args.tensor_parallel} is not a power of 2; "
            "uncommon for real deployments but allowed"
        )

    if args.cache_dir:
        cache_path = Path(args.cache_dir).resolve()
        cache_path_str = str(cache_path)

        for prefix in FORBIDDEN_CACHE_PREFIXES:
            if cache_path_str.startswith(prefix):
                errors.append(f"--cache-dir cannot be a system directory: {cache_path}")
                break

        parent = cache_path.parent
        if parent.exists() and not os.access(parent, os.W_OK):
            errors.append(f"--cache-dir parent is not writable: {parent}")

    if errors:
        for error in errors:
            logger.error(f"Validation error: {error}")
        raise ValueError(f"Invalid arguments:\n  " + "\n  ".join(errors))

    return args


# ============================================================================
# USER SIMULATION AND WORKLOAD GENERATION
# ============================================================================

class UserSimulator:
    """Generates realistic user workloads based on pre-defined templates."""

    DEFAULT_USER_TEMPLATES = {
        'chatbot': {
            'context_range': (512, 4096), 'generation_range': (50, 200), 'think_time_range': (0.1, 0.5),
        },
        'coding': {
            'context_range': (4096, 25000), 'generation_range': (100, 500), 'think_time_range': (0.2, 1.0),
        },
        'document': {
            'context_range': (4096, 16384), 'generation_range': (200, 800), 'think_time_range': (0.3, 1.5),
        },
    }

    @classmethod
    def _get_user_templates(cls) -> Dict:
        """Get user templates from config, falling back to defaults."""
        templates = {}
        for user_type in ['chatbot', 'coding', 'document']:
            default = cls.DEFAULT_USER_TEMPLATES[user_type]
            templates[user_type] = {
                'context_range': tuple(cfg('user_templates', user_type, 'context_range', default=list(default['context_range']))),
                'generation_range': tuple(cfg('user_templates', user_type, 'generation_range', default=list(default['generation_range']))),
                'think_time_range': tuple(cfg('user_templates', user_type, 'think_time_range', default=list(default['think_time_range']))),
            }
        return templates

    @classmethod
    def generate_user(cls, user_id: str, user_type: str = 'chatbot', priority: int = 1,
                      qos_level: QoSLevel = QoSLevel.BATCH) -> UserProfile:
        """Generates a single user profile based on a template."""
        templates = cls._get_user_templates()
        template = templates.get(user_type, templates['chatbot'])
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
        interactive_prob = cfg('qos_distribution', 'interactive_probability', default=0.15)
        responsive_threshold = cfg('qos_distribution', 'responsive_threshold', default=0.50)

        users = []
        for i in range(num_users):
            user_type = random.choice(['chatbot', 'coding', 'document'])

            rand = random.random()
            if rand < interactive_prob:
                qos_level, priority = QoSLevel.INTERACTIVE, 3
            elif rand < responsive_threshold:
                qos_level, priority = QoSLevel.RESPONSIVE, 2
            else:
                qos_level, priority = QoSLevel.BATCH, 1

            users.append(cls.generate_user(f"user_{i:04d}", user_type, priority, qos_level))
        return users


# ============================================================================
# SHAREGPT DATASET LOADER
# ============================================================================

class ShareGPTDatasetLoader:
    """
    Loads ShareGPT conversation data and provides realistic request patterns.
    """

    def __init__(self, dataset_path: str, max_conversations: int = 1000, seed: Optional[int] = None):
        self.dataset_path = dataset_path
        self.max_conversations = max_conversations
        self.conversations = []
        self.token_stats = {}

        if seed:
            random.seed(seed)
            np.random.seed(seed)

        self._load_dataset()

    def _load_dataset(self):
        """Load and process the ShareGPT dataset."""
        if not os.path.exists(self.dataset_path):
            logger.warning(f"Dataset not found at {self.dataset_path}")
            return

        try:
            tokenizer = None
            if TIKTOKEN_AVAILABLE:
                try:
                    tokenizer = tiktoken.get_encoding("cl100k_base")
                except Exception:
                    pass

            if tokenizer is None:
                logger.info("Tiktoken not available, using approximate token counting")

            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            for conv_idx, conversation in enumerate(data[:self.max_conversations]):
                if 'conversations' not in conversation:
                    continue

                conv_data = []
                turns = conversation['conversations']

                for i in range(0, len(turns) - 1, 2):
                    if i + 1 >= len(turns):
                        break

                    human_turn = turns[i]
                    gpt_turn = turns[i + 1]

                    if human_turn.get('from') != 'human' or gpt_turn.get('from') != 'gpt':
                        continue

                    context_text = human_turn.get('value', '')
                    generation_text = gpt_turn.get('value', '')

                    if tokenizer:
                        context_tokens = len(tokenizer.encode(context_text))
                        generation_tokens = len(tokenizer.encode(generation_text))
                    else:
                        context_tokens = max(1, len(context_text) // 4)
                        generation_tokens = max(1, len(generation_text) // 4)

                    context_tokens = min(context_tokens, 16384)
                    generation_tokens = min(generation_tokens, 2048)

                    conv_data.append({
                        'context_tokens': context_tokens,
                        'generation_tokens': generation_tokens,
                        'turn_number': i // 2 + 1
                    })

                if conv_data:
                    self.conversations.append({
                        'id': conversation.get('id', f'conv_{conv_idx}'),
                        'turns': conv_data
                    })

            if self.conversations:
                all_context_tokens = []
                all_generation_tokens = []

                for conv in self.conversations:
                    for turn in conv['turns']:
                        all_context_tokens.append(turn['context_tokens'])
                        all_generation_tokens.append(turn['generation_tokens'])

                self.token_stats = {
                    'context_mean': np.mean(all_context_tokens),
                    'context_std': np.std(all_context_tokens),
                    'context_min': np.min(all_context_tokens),
                    'context_max': np.max(all_context_tokens),
                    'context_p50': np.percentile(all_context_tokens, 50),
                    'context_p95': np.percentile(all_context_tokens, 95),
                    'generation_mean': np.mean(all_generation_tokens),
                    'generation_std': np.std(all_generation_tokens),
                    'generation_min': np.min(all_generation_tokens),
                    'generation_max': np.max(all_generation_tokens),
                    'generation_p50': np.percentile(all_generation_tokens, 50),
                    'generation_p95': np.percentile(all_generation_tokens, 95),
                    'total_conversations': len(self.conversations),
                    'total_turns': sum(len(c['turns']) for c in self.conversations)
                }

                logger.info(f"Loaded {len(self.conversations)} conversations with {self.token_stats['total_turns']} turns")
                logger.info(f"Context tokens: mean={self.token_stats['context_mean']:.1f}, p50={self.token_stats['context_p50']:.1f}, p95={self.token_stats['context_p95']:.1f}")
                logger.info(f"Generation tokens: mean={self.token_stats['generation_mean']:.1f}, p50={self.token_stats['generation_p50']:.1f}, p95={self.token_stats['generation_p95']:.1f}")

        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            self.conversations = []

    def get_random_conversation(self) -> Optional[Dict]:
        """Get a random conversation from the dataset."""
        if not self.conversations:
            return None
        return random.choice(self.conversations)

    def get_random_turn(self) -> Optional[Tuple[int, int]]:
        """Get random context and generation token counts from the dataset."""
        if not self.conversations:
            return None

        conv = self.get_random_conversation()
        if conv and conv['turns']:
            turn = random.choice(conv['turns'])
            return turn['context_tokens'], turn['generation_tokens']
        return None

    def iterate_conversations(self, shuffle: bool = True):
        """Iterate through all conversations, optionally shuffled."""
        conversations = self.conversations.copy()
        if shuffle:
            random.shuffle(conversations)
        for conv in conversations:
            yield conv
