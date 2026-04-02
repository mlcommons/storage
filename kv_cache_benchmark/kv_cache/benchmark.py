"""
Integrated benchmark orchestrator for KV Cache Benchmark.

Contains IntegratedBenchmark which wires all components together
and runs the main benchmark loop with thread management, trace replay,
preconditioning, and summary printing.
"""

import os
import sys
import csv
import glob
import time
import queue
import signal
import random
import logging
import threading
import subprocess
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from kv_cache.config import cfg
from kv_cache.models import (
    ModelConfig, InferencePhase, GenerationMode, GENERATION_TIMING,
    QoSLevel, QOS_PROFILES, UserProfile, InferenceRequest,
)
from kv_cache.cache import MultiTierCache
from kv_cache.conversation import ConversationManager
from kv_cache.prefix_cache import PrefixType, PrefixCacheManager
from kv_cache.rag import RAGDocumentManager
from kv_cache.monitoring import StorageMonitor, WorkloadAutoscaler, QoSMonitor
from kv_cache.workload import (
    ValidationEngine, UserSimulator, ShareGPTDatasetLoader,
)
from kv_cache.tracer import IOTracer

logger = logging.getLogger(__name__)


class IntegratedBenchmark:
    """The main orchestrator for the entire benchmark."""

    def __init__(self,
                 model_config: ModelConfig,
                 num_users: int,
                 gpu_memory_gb: float,
                 cpu_memory_gb: float,
                 duration_seconds: int,
                 num_gpus: int = 1,
                 tensor_parallel: int = 1,
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
                 dataset_path: Optional[str] = None,
                 max_conversations: int = 500,
                 seed: Optional[int] = None,
                 max_concurrent_allocs: int = 0,
                 request_rate: float = 0,
                 max_requests: int = 0,
                 storage_capacity_gb: float = 0,
                 precondition: bool = False,
                 precondition_size_gb: float = 0,
                 precondition_threads: int = 0,
                 trace_speedup: float = 1.0,
                 replay_cycles: int = 0,
                 prefill_only: bool = False,
                 decode_only: bool = False,
                 io_trace_log: Optional[str] = None,
                 enable_latency_tracing: bool = False):

        self.model_config = model_config
        self.num_users = num_users
        self.initial_users = num_users
        self.duration = duration_seconds
        self.num_gpus = max(1, num_gpus)
        self.tensor_parallel = max(1, tensor_parallel)
        self.gpu_memory_gb_per_card = gpu_memory_gb
        self.total_gpu_memory_gb = gpu_memory_gb * self.num_gpus
        self.enable_autoscaling = enable_autoscaling
        self.enable_multi_turn = enable_multi_turn
        self.enable_latency_tracing = enable_latency_tracing
        self._trace_proc = None
        self.generation_mode = generation_mode
        self.ms_per_token = GENERATION_TIMING[generation_mode] * 1000
        self.enable_prefix_caching = enable_prefix_caching
        self.enable_rag = enable_rag
        self.rag_num_docs = rag_num_docs
        self.performance_profile = performance_profile
        self.use_burst_trace = use_burst_trace
        self.burst_trace_path = burst_trace_path
        self.dataset_path = dataset_path
        self.max_conversations = max_conversations
        self.seed = seed
        self.max_concurrent_allocs = max_concurrent_allocs
        self.request_rate = request_rate
        self.max_requests = max_requests
        self.storage_capacity_gb = storage_capacity_gb
        self.precondition = precondition
        self.precondition_size_gb = precondition_size_gb
        self.precondition_threads = precondition_threads if precondition_threads > 0 else (os.cpu_count() or 4)
        self.trace_speedup = trace_speedup
        self.replay_cycles = replay_cycles
        self.prefill_only = prefill_only
        self.decode_only = decode_only

        # Trace mode: IOTracer is created here and closed at the end of run()
        if io_trace_log:
            self.io_tracer: Optional[IOTracer] = IOTracer(io_trace_log)
        else:
            self.io_tracer = None
        self.burst_trace_files: List[str] = []
        self.sharegpt_loader: Optional[ShareGPTDatasetLoader] = None

        if self.dataset_path:
            self.sharegpt_loader = ShareGPTDatasetLoader(
                dataset_path=self.dataset_path,
                max_conversations=self.max_conversations,
                seed=self.seed
            )
            self.use_dataset = True
        elif self.use_burst_trace:
            self.burst_trace_files = self._resolve_burst_trace_files()
            self.use_dataset = False
        else:
            self.use_dataset = False

        # Initialize components
        self.cache = MultiTierCache(
            model_config=model_config,
            gpu_memory_gb=self.total_gpu_memory_gb,
            cpu_memory_gb=cpu_memory_gb,
            cache_dir=cache_dir,
            performance_profile=performance_profile,
            seed=seed,
            max_concurrent_allocs=max_concurrent_allocs,
            storage_capacity_gb=storage_capacity_gb,
            tensor_parallel=self.tensor_parallel,
            io_tracer=self.io_tracer,
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

        self.results = {
            'requests_completed': 0, 'total_tokens_generated': 0,
            'total_storage_io_latency': 0.0, 'total_generation_latency': 0.0,
            'end_to_end_latencies': [], 'storage_latencies': [], 'generation_latencies': [],
            'throughput_timeline': [], 'prefill_latencies': [], 'decode_latencies': [],
            'multi_turn_cache_hits': 0, 'multi_turn_cache_misses': 0,
            'seed': self.seed,
        }
        self.results_lock = threading.Lock()
        self.stop_event: Optional[threading.Event] = None
        self.rag_ingest_done = threading.Event() if self.enable_rag else None

    def _ingest_rag_documents(self, num_docs: int, stop_event: Optional[threading.Event] = None):
        """Ingests RAG documents for the workload."""
        logger.info(f"Ingesting {num_docs} RAG documents...")

        # Determine token range based on model size
        # Large models (70B+) have bigger per-token KV cache, so use fewer tokens per doc
        is_large_model = self.model_config.hidden_dim >= 8192 or self.model_config.num_layers >= 64
        if is_large_model:
            token_min = cfg('rag', 'large_model_doc_tokens_min', default=1024)
            token_max = cfg('rag', 'large_model_doc_tokens_max', default=4096)
        else:
            token_min = cfg('rag', 'small_model_doc_tokens_min', default=4000)
            token_max = cfg('rag', 'small_model_doc_tokens_max', default=12000)

        logger.info(f"RAG document token range: [{token_min}, {token_max}] "
                    f"({'large' if is_large_model else 'small'} model profile)")

        for i in range(num_docs):
            if stop_event and stop_event.is_set():
                break
            doc_tokens = random.randint(token_min, token_max)
            self.rag_manager.ingest_document(f"doc_{i:04d}", doc_tokens, self.model_config)

        if self.rag_ingest_done:
            self.rag_ingest_done.set()

    def _resolve_burst_trace_files(self) -> List[str]:
        """Resolve --burst-trace-path to a sorted list of CSV file paths."""
        p = self.burst_trace_path
        if not p:
            logger.error("--use-burst-trace flag requires --burst-trace-path to be set.")
            sys.exit(1)

        if os.path.isdir(p):
            files = sorted(glob.glob(os.path.join(p, '*.csv')))
        elif '*' in p or '?' in p:
            files = sorted(glob.glob(p))
        elif os.path.isfile(p):
            files = [p]
        else:
            logger.error(f"Trace path not found: {p}")
            sys.exit(1)

        if not files:
            logger.error(f"No CSV files matched: {p}")
            sys.exit(1)

        logger.info(f"Resolved {len(files)} BurstGPT trace file(s): {[os.path.basename(f) for f in files]}")
        return files

    def _burst_trace_iterator(self):
        """Streaming iterator that yields trace rows from each CSV file."""
        for filepath in self.burst_trace_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            timestamp = float(row.get('Timestamp', 0))
                            context_tokens = int(row['Request tokens'])
                            generate_tokens = int(row['Response tokens'])
                            total_tokens = int(row.get('Total tokens', context_tokens + generate_tokens))
                            yield (timestamp, context_tokens, generate_tokens, total_tokens)
                        except (ValueError, KeyError):
                            continue
            except FileNotFoundError:
                logger.error(f"Trace file not found: {filepath}")
                sys.exit(1)
            except Exception as e:
                logger.error(f"Error reading trace file {filepath}: {e}")
                sys.exit(1)

    def _generate_requests_from_trace(self, stop_event: threading.Event):
        """Generates InferenceRequest objects from the streaming trace iterator."""
        speedup = self.trace_speedup
        cycles_remaining = self.replay_cycles
        request_index = 0
        prev_timestamp = None
        trace_total_tokens_sum = 0

        interactive_prob = cfg('qos_distribution', 'interactive_probability', default=0.15)
        responsive_threshold = cfg('qos_distribution', 'responsive_threshold', default=0.50)

        while not stop_event.is_set():
            rows_in_cycle = 0
            for timestamp, context_tokens, generate_tokens, total_tokens in self._burst_trace_iterator():
                if stop_event.is_set():
                    break

                if prev_timestamp is not None and speedup > 0:
                    delta = timestamp - prev_timestamp
                    if delta > 0:
                        sleep_time = delta / speedup
                        remaining = sleep_time
                        while remaining > 0 and not stop_event.is_set():
                            chunk = min(remaining, 5.0)
                            time.sleep(chunk)
                            remaining -= chunk
                        if stop_event.is_set():
                            break
                prev_timestamp = timestamp

                trace_total_tokens_sum += total_tokens

                with self.counter_lock:
                    req_id = self.request_counter
                    self.request_counter += 1

                rand = random.random()
                if rand < interactive_prob:
                    qos_level, priority = QoSLevel.INTERACTIVE, 3
                elif rand < responsive_threshold:
                    qos_level, priority = QoSLevel.RESPONSIVE, 2
                else:
                    qos_level, priority = QoSLevel.BATCH, 1

                user_id = f"trace_user_{request_index % 1000}"

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
                rows_in_cycle += 1

            if rows_in_cycle == 0:
                logger.warning("BurstGPT trace yielded 0 rows.")
                break

            if cycles_remaining > 0:
                cycles_remaining -= 1
                if cycles_remaining == 0:
                    logger.info(f"Completed {self.replay_cycles} replay cycle(s). "
                                f"Trace total_tokens sum: {trace_total_tokens_sum:,}")
                    if self.stop_event:
                        self.stop_event.set()
                    break

            prev_timestamp = None

    def _generate_requests_from_dataset(self, stop_event: threading.Event):
        """Generates InferenceRequest objects from the loaded ShareGPT dataset."""
        if not self.sharegpt_loader or not self.sharegpt_loader.conversations:
            logger.warning("ShareGPT dataset is empty or not loaded. Falling back to synthetic workload.")
            users = UserSimulator.generate_mixed_users(self.num_users)
            self.generate_requests(users, stop_event)
            return

        conversation_iterator = iter(self.sharegpt_loader.iterate_conversations(shuffle=True))
        current_conversation = None
        turn_index = 0
        cycles_remaining = self.replay_cycles

        while not stop_event.is_set():
            if current_conversation is None or turn_index >= len(current_conversation['turns']):
                try:
                    current_conversation = next(conversation_iterator)
                    turn_index = 0
                except StopIteration:
                    if cycles_remaining > 0:
                        cycles_remaining -= 1
                        if cycles_remaining == 0:
                            logger.info(f"Completed {self.replay_cycles} ShareGPT replay cycle(s).")
                            if self.stop_event:
                                self.stop_event.set()
                            return
                    conversation_iterator = iter(self.sharegpt_loader.iterate_conversations(shuffle=True))
                    continue

            turn = current_conversation['turns'][turn_index]
            context_tokens = turn['context_tokens']
            generate_tokens = turn['generation_tokens']

            with self.counter_lock:
                req_id = self.request_counter
                self.request_counter += 1

            interactive_prob = cfg('qos_distribution', 'interactive_probability', default=0.15)
            responsive_threshold = cfg('qos_distribution', 'responsive_threshold', default=0.50)

            rand = random.random()
            if rand < interactive_prob:
                qos_level, priority = QoSLevel.INTERACTIVE, 3
            elif rand < responsive_threshold:
                qos_level, priority = QoSLevel.RESPONSIVE, 2
            else:
                qos_level, priority = QoSLevel.BATCH, 1

            user_id = f"dataset_user_{req_id % self.num_users}"
            conv_id = current_conversation['id']

            phase = InferencePhase.PREFILL if context_tokens >= 10000 else InferencePhase.PREFILL_DECODE

            request = InferenceRequest(
                user_id=user_id,
                request_id=f"{user_id}_req_{req_id:04d}",
                timestamp=datetime.now(),
                context_tokens=context_tokens,
                generate_tokens=generate_tokens,
                priority=priority,
                phase=phase,
                qos_level=qos_level,
                cache_key=f"{conv_id}_turn_{turn['turn_number']}",
                conversation_id=conv_id if self.enable_multi_turn else None,
                turn_number=turn['turn_number'] if self.enable_multi_turn else None
            )

            priority_tuple = (-QOS_PROFILES[request.qos_level].priority, time.time())
            self.request_queue.put((priority_tuple, request))

            turn_index += 1

            if self.request_rate > 0:
                time.sleep(1.0 / self.request_rate)

    def generate_requests(self, users: List[UserProfile], stop_event: threading.Event):
        """Generate requests concurrently for each simulated user."""

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
                time.sleep(user.think_time * random.uniform(0.8, 1.2))
                if stop_event.is_set():
                    break

                if self.enable_multi_turn and self.conversation_manager:
                    if local_conv_id and random.random() >= 0.8:
                        with self.user_conversations_lock:
                            self.user_conversations.pop(user.user_id, None)
                        local_conv_id = None

                    # Enforce max_turns_per_conv hard cap
                    if local_conv_id:
                        with self.conversation_manager.lock:
                            state = self.conversation_manager.conversations.get(local_conv_id)
                            if state and state.turn_number >= self.conversation_manager.max_turns_per_conv:
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

                if self.rag_manager and random.random() < cfg('rag', 'request_probability', default=0.1):
                    doc_keys = list(self.rag_manager.documents.keys())
                    if not doc_keys:
                        continue  # RAG documents not yet ingested
                    doc_id = random.choice(doc_keys)
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

        for user in users:
            threading.Thread(target=user_worker, args=(user,), daemon=True).start()

        self.active_users = users

        stop_event.wait()

    def process_requests(self, stop_event: threading.Event):
        """The main worker loop that processes requests from the queue."""
        while not stop_event.is_set():
            try:
                priority_tuple, request = self.request_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            # Check again after dequeue — don't start expensive I/O after stop
            if stop_event.is_set():
                break

            request.start_time = time.perf_counter()
            storage_latency = 0.0
            cache_type = 'user'

            # 1. Check for a prefix cache hit.
            if self.prefix_cache_manager:
                prefix_entry, remaining_tokens = self.prefix_cache_manager.check_prefix_cache(request, self.model_config)
                if prefix_entry:
                    cache_type = 'system' if prefix_entry.prefix_type == PrefixType.SYSTEM_PROMPT else 'common'
                    _, read_lat = self.cache.access_cache(prefix_entry.kv_cache_key, request.phase, cache_type)
                    storage_latency += read_lat
                    request.context_tokens = remaining_tokens

            # Skip steps 2+3 entirely in decode_only mode:
            # - Step 2 reads always miss (step 3 writes are skipped, so no entries exist)
            # - Step 3 prefill writes don't apply to decode-only nodes
            if not self.decode_only:
                # 2. For multi-turn conversations, reload all previous turns' KV cache.
                #    Reads every previous turn via access_cache (real I/O for entries
                #    that survived eviction; immediate (None, 0.0) for evicted entries).
                if self.conversation_manager and request.turn_number > 1:
                    prev_keys = self.conversation_manager.get_all_previous_turn_keys(
                        request.conversation_id, request.turn_number
                    )
                    for prev_turn_key in prev_keys:
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

            # 4. Simulate a RAG operation.
            if self.rag_manager and random.random() < cfg('rag', 'request_probability', default=0.1):
                doc_keys = list(self.rag_manager.documents.keys()) if self.rag_manager.documents else []
                if doc_keys:
                    doc_id = random.choice(doc_keys)
                    chunks = self.rag_manager.retrieve_chunks(doc_id)
                    for chunk in chunks:
                        _, read_lat = self.cache.access_cache(chunk.kv_cache_key, InferencePhase.DECODE)
                        storage_latency += read_lat

            # 5. Perform the DECODE operation (a cache READ).
            # Skip if prefill_only mode (disaggregated prefill node)
            if not self.prefill_only:
                if request.phase == InferencePhase.DECODE or request.phase == InferencePhase.PREFILL_DECODE:
                    # For decode-only mode, read from pre-populated cache entries
                    if self.decode_only and hasattr(self, '_prepopulated_keys') and self._prepopulated_keys:
                        # Pick a random pre-populated key to read from
                        decode_key = random.choice(self._prepopulated_keys)
                    else:
                        decode_key = request.cache_key
                    
                    location, read_latency = self.cache.access_cache(decode_key, InferencePhase.DECODE, cache_type)
                    storage_latency += read_latency
                    decode_total_latency = read_latency

                    if location is None:
                        # Cache miss during decode - need to allocate (unless decode_only)
                        if not self.decode_only:
                            _, _, write_latency = self.cache.allocate_cache(
                                request.cache_key,
                                request.context_tokens,
                                InferencePhase.PREFILL
                            )
                            storage_latency += write_latency
                    else:
                        decode_batch_size = cfg('decode', 'batch_size', default=32)
                        num_batched_reads = max(1, (request.generate_tokens + decode_batch_size - 1) // decode_batch_size)
                        for _ in range(num_batched_reads):
                            _, batch_read_latency = self.cache.access_cache(decode_key, InferencePhase.DECODE, cache_type)
                            storage_latency += batch_read_latency
                            decode_total_latency += batch_read_latency

                    with self.results_lock: self.results['decode_latencies'].append(decode_total_latency)

            # 6. Simulate token generation time.
            generation_latency = request.generate_tokens * GENERATION_TIMING[self.generation_mode]
            if generation_latency > 0: time.sleep(generation_latency)

            request.complete_time = time.perf_counter()

            # 7. Record all results.
            with self.results_lock:
                self.results['requests_completed'] += 1
                self.results['total_tokens_generated'] += request.generate_tokens
                self.results['total_storage_io_latency'] += storage_latency
                self.results['total_generation_latency'] += generation_latency
                self.results['end_to_end_latencies'].append(request.total_latency_ms / 1000)
                self.results['storage_latencies'].append(storage_latency)
                self.results['generation_latencies'].append(generation_latency)

                if self.max_requests > 0 and self.results['requests_completed'] >= self.max_requests:
                    if self.stop_event:
                        self.stop_event.set()

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
                    logger.info(f"Autoscaler {action} -> {self.num_users} users (saturation: {saturation_level:.2f})")
                elif action == 'stop':
                    logger.info("Autoscaler requested stop after reaching capacity peak.")
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
                    self.autoscaler.current_users = self.num_users

            if now - last_log_time >= 10:
                self._calculate_stats()
                queue_depth = self.request_queue.qsize()
                logger.info(f"Time: {int(elapsed)}s, Users: {self.num_users}, Queue: {queue_depth}, "
                      f"Throughput: {throughput:.2f} tok/s")
                last_log_time = now

    def _start_latency_tracing(self):
        """Spawn bpftrace as a sudo subprocess to trace block-layer device latency."""
        script_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'utils')
        script_path = os.path.join(script_dir, 'storage_latency_stack.sh')

        if not os.path.exists(script_path):
            logger.warning(f"Tracing script not found: {script_path}")
            print(f"  WARNING: {script_path} not found. Skipping latency tracing.")
            return

        # Determine the process name to filter on
        comm = os.path.basename(sys.argv[0]) if sys.argv[0] else 'python3'

        print(f"\n### LATENCY TRACING ###")
        print(f"  Script: {script_path}")
        print(f"  Filter: {comm}")
        print(f"  Spawning sudo bpftrace (you may be prompted for password)...")

        try:
            self._trace_proc = subprocess.Popen(
                ['sudo', script_path, comm],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                preexec_fn=os.setsid  # own process group for clean SIGINT
            )
            # Give bpftrace time to attach probes
            time.sleep(2)
            if self._trace_proc.poll() is not None:
                stderr = self._trace_proc.stderr.read().decode('utf-8', errors='replace')
                logger.warning(f"bpftrace exited early: {stderr[:200]}")
                print(f"  WARNING: bpftrace failed to start. Continuing without tracing.")
                self._trace_proc = None
            else:
                print(f"  bpftrace attached (pid {self._trace_proc.pid}). Tracing active.")
        except FileNotFoundError:
            logger.warning("sudo or bpftrace not found")
            print(f"  WARNING: sudo/bpftrace not available. Skipping latency tracing.")
            self._trace_proc = None
        except Exception as e:
            logger.warning(f"Failed to start tracing: {e}")
            print(f"  WARNING: Tracing failed: {e}. Continuing without tracing.")
            self._trace_proc = None

    def _stop_latency_tracing(self) -> Optional[Dict]:
        """Send SIGINT to bpftrace, capture histograms, parse into dict."""
        if not self._trace_proc:
            return None

        print(f"\n### COLLECTING TRACE DATA ###")
        try:
            # SIGINT to the process group triggers bpftrace's END block
            os.killpg(os.getpgid(self._trace_proc.pid), signal.SIGINT)
            stdout, stderr = self._trace_proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(self._trace_proc.pid), signal.SIGKILL)
            stdout, stderr = self._trace_proc.communicate()
        except Exception as e:
            logger.warning(f"Error stopping bpftrace: {e}")
            return None

        # bpftrace may print histograms to stdout or stderr depending on version
        stdout_text = stdout.decode('utf-8', errors='replace')
        stderr_text = stderr.decode('utf-8', errors='replace')
        output = stdout_text + '\n' + stderr_text

        if '@' not in output:
            logger.warning(f"No histogram data in bpftrace output ({len(output)} bytes)")
            logger.warning(f"Raw output: {repr(output[:500])}")
            return {}

        return self._parse_bpftrace_output(output)

    def _parse_bpftrace_output(self, output: str) -> Dict:
        """Parse bpftrace histogram output into structured dict."""
        result = {}
        current_hist = None
        lines = output.split('\n')

        for line in lines:
            # Match histogram name like "@d2c_read_us:" or "@vfs_write_us:"
            # bpftrace may prefix with newlines or spaces
            hist_match = re.match(r'^\s*@(\w+):\s*$', line)
            if hist_match:
                current_hist = hist_match.group(1)
                result[current_hist] = {'buckets': [], 'raw': []}
                continue

            # Match histogram bucket like "[128, 256)         5 |@@@@                |"
            bucket_match = re.match(r'^\[(\d+),\s*(\d+)\)\s+(\d+)\s+\|', line)
            if bucket_match and current_hist:
                low = int(bucket_match.group(1))
                high = int(bucket_match.group(2))
                count = int(bucket_match.group(3))
                result[current_hist]['buckets'].append({
                    'range_us': [low, high],
                    'count': count
                })
                result[current_hist]['raw'].append(line.rstrip())
                continue

            # Match single-value bucket like "[1M, 2M)           1 |@@@@                |"
            bucket_match_k = re.match(r'^\[(\d+)([KM]),\s*(\d+)([KM])\)\s+(\d+)\s+\|', line)
            if bucket_match_k and current_hist:
                def parse_val(num, suffix):
                    v = int(num)
                    return v * 1024 if suffix == 'K' else v * 1048576
                low = parse_val(bucket_match_k.group(1), bucket_match_k.group(2))
                high = parse_val(bucket_match_k.group(3), bucket_match_k.group(4))
                count = int(bucket_match_k.group(5))
                result[current_hist]['buckets'].append({
                    'range_us': [low, high],
                    'count': count
                })
                result[current_hist]['raw'].append(line.rstrip())

        return result

    @staticmethod
    def _hist_percentile(buckets: List[Dict], pct: float) -> Dict:
        """Compute a percentile bucket from a parsed histogram."""
        total = sum(b['count'] for b in buckets)
        if total == 0:
            return buckets[0] if buckets else {'range_us': [0, 0], 'count': 0}
        target = total * pct / 100.0
        cumulative = 0
        for b in buckets:
            cumulative += b['count']
            if cumulative >= target:
                return b
        return buckets[-1]

    def _print_trace_results(self, trace_data: Dict):
        """Print parsed bpftrace histograms."""
        if not trace_data:
            return

        print(f"\n### DEVICE LATENCY TRACING (bpftrace) ###")

        # ── Latency histograms ──
        latency_histograms = [
            ('d2c_read_us', 'D2C Read (device hardware time)', 'us'),
            ('d2c_write_us', 'D2C Write (device hardware time)', 'us'),
            ('q2d_read_us', 'Q2D Read (I/O scheduler queue)', 'us'),
            ('q2d_write_us', 'Q2D Write (I/O scheduler queue)', 'us'),
            ('vfs_read_us', 'VFS Read (application-visible)', 'us'),
            ('vfs_write_us', 'VFS Write (application-visible)', 'us'),
            ('fsync_us', 'fsync (device flush)', 'us'),
            ('write_to_fsync_us', 'Write-to-fsync gap (CPU serialization)', 'us'),
            ('fadvise_to_read_us', 'fadvise-to-read gap (cache drop)', 'us'),
        ]

        for key, label, unit in latency_histograms:
            if key not in trace_data or not trace_data[key]['buckets']:
                continue
            buckets = trace_data[key]['buckets']
            total_count = sum(b['count'] for b in buckets)
            if total_count == 0:
                continue

            p50 = self._hist_percentile(buckets, 50)
            p95 = self._hist_percentile(buckets, 95)
            p99 = self._hist_percentile(buckets, 99)

            print(f"\n  {label}:")
            print(f"    Samples: {total_count}")
            print(f"    P50: [{p50['range_us'][0]:,}, {p50['range_us'][1]:,}) {unit}")
            print(f"    P95: [{p95['range_us'][0]:,}, {p95['range_us'][1]:,}) {unit}")
            print(f"    P99: [{p99['range_us'][0]:,}, {p99['range_us'][1]:,}) {unit}")
            for raw_line in trace_data[key]['raw']:
                print(f"    {raw_line}")

        # ── Block size distribution ──
        bssplit_histograms = [
            ('bssplit_read_kb', 'Block Size Distribution (Reads)'),
            ('bssplit_write_kb', 'Block Size Distribution (Writes)'),
        ]

        for key, label in bssplit_histograms:
            if key not in trace_data or not trace_data[key]['buckets']:
                continue
            buckets = trace_data[key]['buckets']
            total_count = sum(b['count'] for b in buckets)
            if total_count == 0:
                continue

            p50 = self._hist_percentile(buckets, 50)

            print(f"\n  {label}:")
            print(f"    I/O count: {total_count}")
            print(f"    P50: [{p50['range_us'][0]:,}, {p50['range_us'][1]:,}) KB")
            for raw_line in trace_data[key]['raw']:
                print(f"    {raw_line}")

        # ── Queue depth distribution ──
        qd_histograms = [
            ('qd_read', 'Queue Depth at Dispatch (Reads)'),
            ('qd_write', 'Queue Depth at Dispatch (Writes)'),
        ]

        for key, label in qd_histograms:
            if key not in trace_data or not trace_data[key]['buckets']:
                continue
            buckets = trace_data[key]['buckets']
            total_count = sum(b['count'] for b in buckets)
            if total_count == 0:
                continue

            p50 = self._hist_percentile(buckets, 50)
            p95 = self._hist_percentile(buckets, 95)

            print(f"\n  {label}:")
            print(f"    Samples: {total_count}")
            print(f"    P50: [{p50['range_us'][0]}, {p50['range_us'][1]})")
            print(f"    P95: [{p95['range_us'][0]}, {p95['range_us'][1]})")
            for raw_line in trace_data[key]['raw']:
                print(f"    {raw_line}")

        # ── LBA heatmap ──
        lba_histograms = [
            ('lba_read_gb', 'LBA Heatmap (Reads, 10 GB buckets)'),
            ('lba_write_gb', 'LBA Heatmap (Writes, 10 GB buckets)'),
        ]

        for key, label in lba_histograms:
            if key not in trace_data or not trace_data[key]['buckets']:
                continue
            buckets = trace_data[key]['buckets']
            total_count = sum(b['count'] for b in buckets)
            if total_count == 0:
                continue

            # Find the hot zone (buckets with > 1% of I/O)
            hot_zones = [b for b in buckets if b['count'] > total_count * 0.01]
            if hot_zones:
                hot_start = hot_zones[0]['range_us'][0]
                hot_end = hot_zones[-1]['range_us'][1]
                hot_pct = sum(b['count'] for b in hot_zones) * 100.0 / total_count
            else:
                hot_start = hot_end = hot_pct = 0

            print(f"\n  {label}:")
            print(f"    I/O count: {total_count}")
            if hot_zones:
                print(f"    Hot zone: {hot_start}-{hot_end} GB ({hot_pct:.0f}% of I/O)")
            for raw_line in trace_data[key]['raw']:
                print(f"    {raw_line}")

    def _generate_fio_workload(self, trace_data: Dict) -> Optional[str]:
        """Generate a fio workload .ini file from bpftrace trace data.

        Distills the traced block-layer I/O pattern into a standalone fio config
        that reproduces the same bssplit, read/write ratio, queue depth, and
        idle time characteristics observed during the benchmark run.
        """
        # ── Validate minimum required histograms ──
        required = ['bssplit_read_kb', 'bssplit_write_kb']
        for key in required:
            if key not in trace_data or not trace_data[key].get('buckets'):
                logger.warning(f"Missing {key} histogram; cannot generate fio workload")
                return None

        # ── bssplit: convert histogram buckets to fio format ──
        def hist_to_bssplit(buckets: List[Dict]) -> str:
            total = sum(b['count'] for b in buckets)
            if total == 0:
                return "4k/100"
            parts = []
            for b in buckets:
                if b['count'] == 0:
                    continue
                size_kb = b['range_us'][0]  # lower bound of bucket
                pct = int(round(b['count'] * 100.0 / total))
                if pct == 0 and b['count'] > 0:
                    pct = 1  # don't drop non-zero buckets
                # Format size: use k for < 1024, m for >= 1024
                if size_kb >= 1024:
                    size_str = f"{size_kb // 1024}m"
                else:
                    size_str = f"{size_kb}k"
                parts.append(f"{size_str}/{pct}")
            return ":".join(parts) if parts else "4k/100"

        read_bssplit = hist_to_bssplit(trace_data['bssplit_read_kb']['buckets'])
        write_bssplit = hist_to_bssplit(trace_data['bssplit_write_kb']['buckets'])
        bssplit_line = f"{read_bssplit},{write_bssplit}"

        # ── rwmixread: from I/O count ratio ──
        read_count = sum(b['count'] for b in trace_data['bssplit_read_kb']['buckets'])
        write_count = sum(b['count'] for b in trace_data['bssplit_write_kb']['buckets'])
        total_io = read_count + write_count
        rwmixread = int(round(read_count * 100.0 / total_io)) if total_io > 0 else 50

        # ── iodepth: from QD histogram P50 ──
        iodepth = 32  # default
        for qd_key in ('qd_read', 'qd_write'):
            if qd_key in trace_data and trace_data[qd_key].get('buckets'):
                p50 = self._hist_percentile(trace_data[qd_key]['buckets'], 50)
                candidate = max(1, p50['range_us'][0])
                iodepth = max(iodepth, candidate)

        # ── thinktime: from write_to_fsync gap (CPU idle between I/O bursts) ──
        thinktime_us = 0
        if 'write_to_fsync_us' in trace_data and trace_data['write_to_fsync_us'].get('buckets'):
            buckets = trace_data['write_to_fsync_us']['buckets']
            total_samples = sum(b['count'] for b in buckets)
            if total_samples >= 4:
                p50 = self._hist_percentile(buckets, 50)
                thinktime_us = p50['range_us'][0]

        # ── thinktime_iotime: from fsync latency (active I/O burst duration) ──
        thinktime_iotime_us = 0
        if 'fsync_us' in trace_data and trace_data['fsync_us'].get('buckets'):
            buckets = trace_data['fsync_us']['buckets']
            total_samples = sum(b['count'] for b in buckets)
            if total_samples >= 4:
                p50 = self._hist_percentile(buckets, 50)
                thinktime_iotime_us = p50['range_us'][0]

        # ── Build the fio config ──
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        model_name = self.model_config.name
        bpt = self.model_config.kv_cache_size_per_token

        lines = [
            f"# KV Cache Benchmark; Traced Workload",
            f"# Generated: {timestamp}",
            f"# Model: {model_name}, Users: {self.num_users}, Duration: {self.duration}s",
            f"# KV bytes/token: {bpt:,} bytes ({bpt/1024:.0f} KB)",
            f"#",
            f"# Distilled from bpftrace block-layer tracing during benchmark run.",
            f"# Total traced I/Os: {total_io:,} ({read_count:,} reads, {write_count:,} writes)",
            f"#",
            f"# Usage:",
            f"#   fio <this_file> --filename=/dev/nvmeXn1",
            f"#   fio <this_file> --filename=/mnt/nvme/fio_test --size=100G",
            f"",
            f"[kv-cache-traced]",
            f"ioengine=libaio",
            f"direct=1",
            f"time_based",
            f"runtime=300",
            f"rw=randrw",
            f"rwmixread={rwmixread}",
            f"bssplit={bssplit_line}",
            f"iodepth={iodepth}",
            f"iodepth_batch_submit={iodepth}",
            f"iodepth_batch_complete_min=1",
            f"size=100%",
        ]

        if thinktime_us > 0:
            lines.extend([
                f"thinktime={thinktime_us}",
                f"thinktime_blocks={iodepth}",
            ])
            if thinktime_iotime_us > 0:
                # thinktime_iotime requires fio 3.28+
                # Converts active I/O period from microseconds to seconds
                thinktime_iotime_s = max(1, thinktime_iotime_us // 1000000)
                lines.append(f"# thinktime_iotime={thinktime_iotime_s}s  # uncomment for fio 3.28+ (active I/O burst before idle)")

        lines.extend([
            f"refill_buffers=1",
            f"norandommap=1",
            f"randrepeat=0",
            f"numjobs=1",
            f"group_reporting",
            f"percentile_list=50:95:99:99.9:99.99",
        ])

        return "\n".join(lines) + "\n"

    def _check_memory_safety(self):
        """Estimate peak memory usage and warn if OOM is likely.

        Peak memory per worker thread comes from access_cache -> np.load + np.array
        which briefly holds 2x the entry size in RAM. With N worker threads running
        concurrently, peak = N * 2 * mean_entry_bytes + baseline (precomputed buffer,
        Python/torch overhead, OS).
        """
        try:
            import psutil
            available_ram = psutil.virtual_memory().available
        except ImportError:
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemAvailable:'):
                            available_ram = int(line.split()[1]) * 1024  # kB to bytes
                            break
                    else:
                        return  # can't determine
            except (FileNotFoundError, ValueError):
                return  # non-Linux or parse error

        bpt = self.model_config.kv_cache_size_per_token
        # Mean context from user templates: midpoint of chatbot + coding + document ranges
        mean_context_tokens = 8000  # conservative estimate
        mean_entry_bytes = mean_context_tokens * bpt
        # np.load + np.array = 2x entry size per concurrent read
        per_thread_peak = mean_entry_bytes * 2
        # Worker threads = min(num_users, 500); all can read concurrently
        num_workers = min(self.num_users, 500)
        # Baseline: precomputed buffer (~256 MB) + Python/torch overhead (~2 GB)
        baseline = 2.5 * 1024**3
        estimated_peak = (num_workers * per_thread_peak) + baseline

        safe_workers = max(1, int((available_ram - baseline) / per_thread_peak)) if per_thread_peak > 0 else num_workers

        print(f"\n### MEMORY SAFETY CHECK ###")
        print(f"  Formula: peak = (workers x 2 x mean_entry_bytes) + baseline")
        print(f"         = ({num_workers} x 2 x {mean_entry_bytes / 1024**2:.0f} MB) + {baseline / 1024**3:.1f} GB")
        print(f"         = {estimated_peak / 1024**3:.1f} GB")
        print(f"  Available RAM: {available_ram / 1024**3:.1f} GB")
        print(f"  Mean entry size: {mean_entry_bytes / 1024**2:.0f} MB  ({mean_context_tokens} tok x {bpt:,} B/tok)")
        print(f"  Peak per thread: {per_thread_peak / 1024**2:.0f} MB  (np.load + np.array copy)")
        print(f"  Worker threads: {num_workers}")
        print(f"  Safe concurrent readers: ~{safe_workers}  = (available - baseline) / peak_per_thread")

        if estimated_peak > available_ram * 0.85:
            print(f"  WARNING: Estimated peak memory ({estimated_peak / 1024**3:.1f} GB) exceeds 85% of")
            print(f"  available RAM ({available_ram / 1024**3:.1f} GB). Risk of OOM with {num_workers} workers.")
            print(f"  Consider: --num-users {min(safe_workers, self.num_users)} or --max-concurrent-allocs {max(1, safe_workers // 2)}")
        else:
            print(f"  Status: OK")

    def run(self) -> Dict:
        """The main entry point to start the benchmark execution."""
        print(f"\nIntegrated Multi-User KV Cache Benchmark - MLPerf Edition")
        print(f"Model: {self.model_config.name}")
        if self.num_gpus > 1 or self.tensor_parallel > 1:
            print(f"System: {self.num_gpus}× {self.gpu_memory_gb_per_card:.0f} GB GPU  "
                  f"(total {self.total_gpu_memory_gb:.0f} GB HBM)  │  TP={self.tensor_parallel}")
        else:
            print(f"GPU Memory: {self.total_gpu_memory_gb:.0f} GB")
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
        if self.io_tracer is not None:
            print(f"  - I/O TRACE MODE: ACTIVE — writing trace to {self.io_tracer.path}")
            print(f"    (No real GPU/CPU/NVMe I/O will be performed)")
        if self.use_burst_trace:
            print(f"    Trace files: {len(self.burst_trace_files)}")
            print(f"    Trace speedup: {self.trace_speedup}x ({'no delay' if self.trace_speedup == 0 else 'real-time' if self.trace_speedup == 1.0 else f'{self.trace_speedup}x faster'})")
            print(f"    Replay cycles: {'infinite' if self.replay_cycles == 0 else self.replay_cycles}")
        print(f"  - ShareGPT Dataset: {'Enabled' if self.use_dataset else 'Disabled'}")
        if self.max_concurrent_allocs > 0:
            print(f"  - Max Concurrent Allocations: {self.max_concurrent_allocs} (bounds RAM usage)")
        print("=" * 80)

        users = []
        if not self.use_burst_trace and not self.use_dataset:
            users = UserSimulator.generate_mixed_users(self.num_users)
            context_lengths = [u.context_length for u in users]
            bytes_per_token_per_rank = self.model_config.kv_cache_size_per_token / self.tensor_parallel
            tp_note = f" per TP rank (full={bytes_per_token_per_rank * self.tensor_parallel / 1024**2 * min(context_lengths):.2f} MB)" if self.tensor_parallel > 1 else ""
            print(f"\nUser Context Length Distribution:")
            print(f"  Min: {min(context_lengths)} tokens ({min(context_lengths) * bytes_per_token_per_rank / 1024**2:.2f} MB{tp_note})")
            print(f"  Max: {max(context_lengths)} tokens ({max(context_lengths) * bytes_per_token_per_rank / 1024**2:.2f} MB)")
            print(f"  Mean: {np.mean(context_lengths):.0f} tokens ({np.mean(context_lengths) * bytes_per_token_per_rank / 1024**2:.2f} MB)")
            if self.tensor_parallel > 1:
                print(f"  (sizes shown are per-rank 1/{self.tensor_parallel} shard; TP={self.tensor_parallel})")

            qos_dist = {level: sum(1 for u in users if u.qos_level == level) for level in QoSLevel}
            print(f"\nQoS Distribution:")
            for level, count in qos_dist.items():
                print(f"  {level.value}: {count} users")
        elif self.use_dataset and self.sharegpt_loader:
            print(f"\nShareGPT Dataset Statistics:")
            print(f"  Conversations: {self.sharegpt_loader.token_stats.get('total_conversations', 0)}")
            print(f"  Total Turns: {self.sharegpt_loader.token_stats.get('total_turns', 0)}")

        self._check_memory_safety()

        if self.precondition:
            self._run_preconditioning()

        # Pre-populate cache for decode-only mode
        if self.decode_only:
            self._prepopulate_cache_for_decode()

        # Log disaggregated mode
        mode_str = "standard (prefill+decode)"
        if self.prefill_only:
            mode_str = "PREFILL-ONLY (write-heavy, disaggregated prefill node)"
        elif self.decode_only:
            mode_str = "DECODE-ONLY (read-heavy, assumes KV cache pre-populated)"
        if self.enable_latency_tracing:
            self._start_latency_tracing()

        print(f"\nStarting benchmark... Mode: {mode_str}")
        print("-" * 80)

        stop_event = threading.Event()
        self.stop_event = stop_event

        threads = []
        if self.use_dataset:
            gen_thread = threading.Thread(target=self._generate_requests_from_dataset, args=(stop_event,), daemon=True)
        elif self.use_burst_trace:
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

        if self.enable_autoscaling:
            mon_thread = threading.Thread(target=self.monitor_stats, args=(stop_event,), daemon=True)
            threads.append(mon_thread)
            mon_thread.start()

        benchmark_start = time.time()
        stop_event.wait(timeout=self.duration)
        actual_duration = time.time() - benchmark_start

        stop_event.set()
        for thread in threads:
            thread.join(timeout=2.0)

        # Stop tracing and collect results before stats calculation
        trace_data = None
        if self.enable_latency_tracing:
            trace_data = self._stop_latency_tracing()

        self._calculate_stats(actual_duration)

        if trace_data:
            self.results['device_latency_tracing'] = trace_data
            self._print_trace_results(trace_data)

            fio_config = self._generate_fio_workload(trace_data)
            if fio_config:
                self.results['fio_workload'] = fio_config
                print(f"\n### GENERATED FIO WORKLOAD ###")
                for line in fio_config.strip().split('\n'):
                    print(f"  {line}")

        if self.validator:
            self.results['validation'] = self.validator.validate_benchmark(self.results)

        if self.io_tracer is not None:
            self.io_tracer.close()

        return self.results

    def _run_preconditioning(self):
        """Run multi-threaded SSD preconditioning phase."""
        nvme_limit = self.cache.nvme_memory_limit
        if self.precondition_size_gb > 0:
            target_bytes = self.precondition_size_gb * 1024**3
        elif nvme_limit != float('inf'):
            target_bytes = 2 * nvme_limit
        else:
            print("WARNING: Cannot precondition — NVMe capacity unknown and --precondition-size-gb not set. Skipping.")
            return

        target_gb = target_bytes / 1024**3
        num_threads = self.precondition_threads
        print(f"\n### PRECONDITIONING PHASE ###")
        print(f"  Target: {target_gb:.1f} GB")
        print(f"  Threads: {num_threads}")

        tokens_per_entry = 2048
        lock = threading.Lock()
        state = {'written_bytes': 0, 'seq': 0, 'last_report': 0}

        def worker():
            consecutive_failures = 0
            while True:
                with lock:
                    if state['written_bytes'] >= target_bytes:
                        return
                    my_seq = state['seq']
                    state['seq'] += 1

                key = f"precond_{my_seq}"
                success, tier, latency = self.cache.allocate_cache(key, tokens_per_entry)

                if success:
                    consecutive_failures = 0
                    entry = self.cache.cache_entries.get(key)
                    if entry:
                        with lock:
                            state['written_bytes'] += entry['size']
                            gb_written = state['written_bytes'] / 1024**3
                            if gb_written - state['last_report'] >= 10:
                                print(f"  Preconditioning progress: {gb_written:.1f} / {target_gb:.1f} GB")
                                state['last_report'] = gb_written
                else:
                    consecutive_failures += 1
                    if consecutive_failures > 50:
                        with lock:
                            print(f"  WARNING: Preconditioning stalled at {state['written_bytes']/1024**3:.1f} GB — filesystem full. Continuing.")
                        return
                    time.sleep(0.1)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker) for _ in range(num_threads)]
            for f in futures:
                f.result()

        print(f"  Preconditioning complete: {state['written_bytes'] / 1024**3:.1f} GB written")
        print(f"  Resetting stats for steady-state measurement...")
        self.cache.reset_stats()

    def _prepopulate_cache_for_decode(self):
        """Pre-populate cache entries for decode-only mode.
        
        In disaggregated inference, the decode node assumes KV cache already exists
        (written by prefill nodes). This simulates that by writing entries upfront.
        """
        print(f"\n### PRE-POPULATING CACHE FOR DECODE-ONLY MODE ###")
        
        # Determine how many entries to pre-populate based on num_users and typical context
        num_entries = self.num_users * 10  # 10 entries per user (multi-turn)
        tokens_per_entry = 2048  # Average context length
        num_threads = os.cpu_count() or 16
        
        print(f"  Creating {num_entries} cache entries ({tokens_per_entry} tokens each)...")
        print(f"  Threads: {num_threads}")
        
        # Temporarily disable semaphore for fast pre-population
        # (pre-population is not part of measured benchmark)
        original_semaphore = self.cache.allocation_semaphore
        self.cache.allocation_semaphore = None
        
        # Track pre-populated keys so decode requests can use them
        self._prepopulated_keys = []
        lock = threading.Lock()
        state = {'completed': 0, 'seq': 0}
        
        def worker():
            while True:
                with lock:
                    if state['seq'] >= num_entries:
                        return
                    my_seq = state['seq']
                    state['seq'] += 1
                
                key = f"prepop_{my_seq}"
                success, tier, latency = self.cache.allocate_cache(key, tokens_per_entry)
                
                with lock:
                    if success:
                        self._prepopulated_keys.append(key)
                    state['completed'] += 1
                    if state['completed'] % 100 == 0:
                        print(f"  Progress: {state['completed']}/{num_entries} entries created")
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(worker) for _ in range(num_threads)]
            for f in futures:
                f.result()
        
        # Restore semaphore for actual benchmark
        self.cache.allocation_semaphore = original_semaphore
        
        print(f"  Pre-population complete: {len(self._prepopulated_keys)} entries in cache")
        print(f"  Resetting stats for decode-only measurement...")
        self.cache.reset_stats()

    def _calculate_stats(self, actual_duration: float = None):
        """Calculate final statistics with all feature breakdowns."""
        if not self.results['end_to_end_latencies']:
            logger.warning("No requests completed during benchmark!")
            return

        duration = actual_duration if actual_duration else self.duration

        e2e = np.array(self.results['end_to_end_latencies'])
        storage = np.array(self.results['storage_latencies'])
        generation = np.array(self.results['generation_latencies'])

        cache_stats = self.cache.get_stats(duration)
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
            'elapsed_time': duration,
            'avg_throughput_tokens_per_sec': self.results['total_tokens_generated'] / duration,
            'total_storage_io_time': self.results['total_storage_io_latency'],
            'storage_throughput_tokens_per_sec': self.results['total_tokens_generated'] / self.results['total_storage_io_latency'] if self.results['total_storage_io_latency'] > 0 else 0,
            'requests_per_second': self.results['requests_completed'] / duration,
            'end_to_end_latency_ms': {
                'mean': np.mean(e2e) * 1000,
                'p50': np.percentile(e2e, 50) * 1000,
                'p95': np.percentile(e2e, 95) * 1000,
                'p99': np.percentile(e2e, 99) * 1000,
                'p999': np.percentile(e2e, 99.9) * 1000,
                'p9999': np.percentile(e2e, 99.99) * 1000,
            },
            'storage_io_latency_ms': {
                'mean': np.mean(storage) * 1000,
                'p50': np.percentile(storage, 50) * 1000,
                'p95': np.percentile(storage, 95) * 1000,
                'p99': np.percentile(storage, 99) * 1000,
                'p999': np.percentile(storage, 99.9) * 1000,
                'p9999': np.percentile(storage, 99.99) * 1000,
            },
            'generation_latency_ms': {
                'mean': np.mean(generation) * 1000,
                'p50': np.percentile(generation, 50) * 1000,
                'p95': np.percentile(generation, 95) * 1000,
                'p99': np.percentile(generation, 99) * 1000,
                'p999': np.percentile(generation, 99.9) * 1000,
                'p9999': np.percentile(generation, 99.99) * 1000,
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
        """Print comprehensive results summary."""
        print("\n" + "=" * 80)
        print("BENCHMARK RESULTS - MLPerf KV Cache Storage Benchmark")
        print(f"Generation Mode: {self.generation_mode.value} ({self.ms_per_token:.1f}ms/token)")
        print("=" * 80)

        # ── KV Block Size Context ──────────────────────────────────────
        # Raised on 2026-03-10 KV Cache TF call: latencies are per entire
        # KV cache block, not per token or per 4 KB page.  Block sizes
        # depend on model architecture and sequence length; they can range
        # from tens of MB to multiple GB.
        bpt = self.model_config.kv_cache_size_per_token
        print(f"\nIMPORTANT: All storage latencies below are measured per KV cache BLOCK,")
        print(f"not per token or per disk page.  Each block holds the full KV state for")
        print(f"one request (all layers, all heads, full sequence length).")
        print(f"  Model KV bytes/token: {bpt:,} bytes ({bpt/1024:.1f} KB)")

        # Compute entry size distribution from live cache entries
        with self.cache.metadata_lock:
            entry_sizes = [e['size'] for e in self.cache.cache_entries.values()]
        if entry_sizes:
            sizes = np.array(entry_sizes)
            print(f"  Entries in cache: {len(sizes)}")
            print(f"  Block size min:   {np.min(sizes)/1024**2:.1f} MB")
            print(f"  Block size mean:  {np.mean(sizes)/1024**2:.1f} MB")
            print(f"  Block size P95:   {np.percentile(sizes, 95)/1024**2:.1f} MB")
            print(f"  Block size max:   {np.max(sizes)/1024**2:.1f} MB")
        else:
            # Fall back to average from aggregate stats
            total_write_bytes = summary.get('cache_stats', {}).get('total_write_bytes', 0)
            write_ops = summary.get('cache_stats', {}).get('write_iops', 0)
            if write_ops > 0:
                avg_mb = (total_write_bytes / write_ops) / 1024**2
                print(f"  Avg block size:   {avg_mb:.1f} MB (from {write_ops} writes)")

        print(f"  A 200 MB block at 1 GB/s NVMe read = ~200 ms device latency.")
        print(f"  Compare latencies against block sizes, not against 4 KB page I/O.\n")

        PASS_SYMBOL = "[OK]"
        FAIL_SYMBOL = "[X]"

        cache_stats = summary['cache_stats']
        if 'storage_health' in cache_stats:
            storage_health = cache_stats['storage_health']
            status = storage_health['overall_status']
            status_symbol = PASS_SYMBOL if status == 'PASS' else FAIL_SYMBOL
            print(f"\n### STORAGE PERFORMANCE ASSESSMENT: {status} {status_symbol} ###")
            print(f"  Criteria Passed: {storage_health['passed_count']}/{storage_health['total_count']}")
            for criterion in storage_health['criteria']:
                symbol = PASS_SYMBOL if criterion['passed'] else FAIL_SYMBOL
                unit = criterion.get('unit', '')
                if unit == 'ratio':
                    print(f"  {symbol} {criterion['name']}: {criterion['actual']:.1%} (target: {criterion['target']:.1%})")
                    continue

                actual = criterion.get('actual')
                target = criterion.get('target')
                try:
                    actual_str = f"{actual:.2f}"
                except (ValueError, TypeError):
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
        print(f"Throughput (wall-clock): {summary['avg_throughput_tokens_per_sec']:.2f} tokens/sec")
        print(f"Throughput (storage I/O): {summary['storage_throughput_tokens_per_sec']:.2f} tokens/sec")
        print(f"Requests/sec: {summary['requests_per_second']:.2f}")

        print(f"\n### END-TO-END LATENCY (Queue Wait + Storage I/O + Generation) ###")
        print(f"  Mean: {summary['end_to_end_latency_ms']['mean']:.2f} ms")
        print(f"  P50:  {summary['end_to_end_latency_ms']['p50']:.2f} ms")
        print(f"  P95:  {summary['end_to_end_latency_ms']['p95']:.2f} ms")
        print(f"  P99:  {summary['end_to_end_latency_ms']['p99']:.2f} ms")

        print(f"\n### PER-REQUEST STORAGE LATENCY (All I/O ops for one request) ###")
        print(f"  Mean: {summary['storage_io_latency_ms']['mean']:.2f} ms")
        print(f"  P50:  {summary['storage_io_latency_ms']['p50']:.2f} ms")
        print(f"  P95:  {summary['storage_io_latency_ms']['p95']:.2f} ms")
        print(f"  P99:  {summary['storage_io_latency_ms']['p99']:.2f} ms")
        print(f"  (= 1 prefill write + N decode reads per request)")

        if self.generation_mode != GenerationMode.NONE:
            print(f"\n### TOKEN GENERATION LATENCY (Simulated @ {self.ms_per_token:.1f}ms/token) ###")
            print(f"  Mean: {summary['generation_latency_ms']['mean']:.2f} ms")
            print(f"  P50:  {summary['generation_latency_ms']['p50']:.2f} ms")
            print(f"  P95:  {summary['generation_latency_ms']['p95']:.2f} ms")

        print(f"\n### STORAGE PERFORMANCE ###")
        print(f"  Cache Hit Rate: {cache_stats['cache_hit_rate']*100:.1f}%")
        print(f"  Total Read: {cache_stats['total_read_gb']:.2f} GB")
        print(f"  Total Write: {cache_stats['total_write_gb']:.2f} GB")
        rw_ratio = cache_stats['read_write_ratio']
        if rw_ratio > 1e9:
            print(f"  Read/Write Ratio: ∞ (read-only)")
        elif rw_ratio < 1e-9:
            print(f"  Read/Write Ratio: 0 (write-only)")
        else:
            print(f"  Read/Write Ratio: {rw_ratio:.2f}")
        print(f"  Storage KV Read Operations/sec: {cache_stats['read_iops'] / self.duration:.2f}")
        print(f"  Storage KV Write Operations/sec: {cache_stats['write_iops'] / self.duration:.2f}")

        print(f"\n### CACHE TIER DISTRIBUTION ###")
        print(f"  GPU Entries: {cache_stats['gpu_entries']} ({cache_stats['gpu_memory_used_gb']:.2f} GB)")
        print(f"  CPU Entries: {cache_stats['cpu_entries']} ({cache_stats['cpu_memory_used_gb']:.2f} GB)")
        print(f"  Storage Entries: {cache_stats['storage_entries']}")

        print(f"\n### TIER-SPECIFIC KV BYTES ###")
        if cache_stats.get('tier_gpu_kv_bytes_written_gb', 0) > 0:
            print(f"  GPU KV Bytes Written: {cache_stats['tier_gpu_kv_bytes_written_gb']:.2f} GB")
        if cache_stats.get('tier_gpu_kv_bytes_read_gb', 0) > 0:
            print(f"  GPU KV Bytes Read: {cache_stats['tier_gpu_kv_bytes_read_gb']:.2f} GB")
        if cache_stats.get('tier_cpu_kv_bytes_written_gb', 0) > 0:
            print(f"  CPU KV Bytes Written: {cache_stats['tier_cpu_kv_bytes_written_gb']:.2f} GB")
        if cache_stats.get('tier_cpu_kv_bytes_read_gb', 0) > 0:
            print(f"  CPU KV Bytes Read: {cache_stats['tier_cpu_kv_bytes_read_gb']:.2f} GB")
        if cache_stats.get('tier_storage_kv_bytes_written_gb', 0) > 0:
            print(f"  Storage KV Bytes Written: {cache_stats['tier_storage_kv_bytes_written_gb']:.2f} GB")
        if cache_stats.get('tier_storage_kv_bytes_read_gb', 0) > 0:
            print(f"  Storage KV Bytes Read: {cache_stats['tier_storage_kv_bytes_read_gb']:.2f} GB")

        print(f"\n### STORAGE KV BANDWIDTH ###")
        for tier_label, tier_key in [('GPU', 'gpu'), ('CPU', 'cpu'), ('Storage', 'storage')]:
            read_bw = cache_stats.get(f'tier_{tier_key}_read_bandwidth_gbps', 0)
            write_bw = cache_stats.get(f'tier_{tier_key}_write_bandwidth_gbps', 0)
            if read_bw > 0:
                print(f"  {tier_label} KV Read Bandwidth: {read_bw:.2f} GB/s")
            if write_bw > 0:
                print(f"  {tier_label} KV Write Bandwidth: {write_bw:.2f} GB/s")

        print(f"\n### TIER-SPECIFIC LATENCIES (Total = Host + Device) ###")
        for tier in ['gpu', 'cpu', 'storage']:
            for op in ['read', 'write']:
                p95_key = f'{tier}_{op}_p95_ms'
                if p95_key in cache_stats:
                    tier_label = 'Storage' if tier == 'storage' else tier.upper()
                    print(f"  {tier_label} {op.title()} P95 (Total): {cache_stats[p95_key]:.2f} ms")

        print(f"\n### STORAGE TIER LATENCY BREAKDOWN (Device = Disk I/O, Host = Serialization) ###")
        for op in ['read', 'write']:
            device_key = f'storage_{op}_device_p95_ms'
            host_key = f'storage_{op}_host_p95_ms'
            total_key = f'storage_{op}_p95_ms'
            if device_key in cache_stats:
                print(f"  Storage {op.title()}:")
                print(f"    - Device P95 (Disk I/O): {cache_stats[device_key]:.2f} ms")
                if host_key in cache_stats:
                    print(f"    - Host P95 (Serialization): {cache_stats[host_key]:.2f} ms")
                if total_key in cache_stats:
                    print(f"    - Total P95: {cache_stats[total_key]:.2f} ms")

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

        if self.performance_profile != 'throughput':
            print(f"\n### QOS LATENCY METRICS (Informational - includes simulated generation) ###")
            qos_metrics = summary['qos_metrics']
            for qos_level, metrics in qos_metrics.items():
                if metrics.get('no_data'): continue
                print(f"\n  {qos_level.upper()}:")
                print(f"    Requests: {metrics['total_requests']}")
                print(f"    Latency P95: {metrics['latency_ms']['p95']:.2f} ms")
                print(f"    Latency P99: {metrics['latency_ms']['p99']:.2f} ms")
                if 'sla' in metrics:
                    sla_met = '[OK]' if metrics['sla']['met'] else '[X]'
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
            print(f"  Validation: {'PASSED [OK]' if validation['passed'] else 'FAILED [X]'}")
            print(f"  Average Error: {validation['avg_error_pct']:.2f}%")

        print("\n" + "=" * 80)
        print("NOTES:")
        if self.generation_mode == GenerationMode.NONE:
            print("  - Pure storage I/O benchmark (no generation simulation)")
        else:
            print("  - End-to-end latency includes simulated GPU inference")
        print("=" * 80)
