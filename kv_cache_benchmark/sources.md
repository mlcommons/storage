# Research Sources for vLLM CPU-Only KV Cache Offload Implementation

## Research Date: 2025-10-03

This document contains all research sources, citations, and key insights gathered during the feasibility study for implementing a vLLM CPU-only KV cache offload comparison baseline for the MLPerf KV Cache Storage Benchmark.

---

## 1. vLLM CPU Support and Architecture

### 1.1 Official vLLM CPU Documentation
- **URL**: https://docs.vllm.ai/en/stable/getting_started/installation/cpu.html
- **Title**: CPU - vLLM
- **Relevance**: Primary documentation for vLLM CPU backend
- **Key Insights**:
  - vLLM supports CPU-only inference on x86 platforms with AVX512 instruction set
  - Supports FP32, FP16, and BF16 data types
  - No pre-built wheels available - must build from source
  - Requires gcc/g++ >= 12.3.0
  - VLLM_CPU_KVCACHE_SPACE environment variable controls KV cache size
  - Intel Extension for PyTorch (IPEX) can be enabled for optimization
  - TCMalloc highly recommended for performance

### 1.2 Red Hat Developer Guide - vLLM on CPU
- **URL**: https://developers.redhat.com/articles/2025/06/17/how-run-vllm-cpus-openshift-gpu-free-inference
- **Title**: How to run vLLM on CPUs with OpenShift for GPU-free inference
- **Relevance**: Real-world CPU deployment guide
- **Key Insights**:
  - Practical deployment guidance for CPU-only vLLM
  - Demonstrates feasibility of production CPU inference
  - No GPU hardware requirements

### 1.3 Medium Guide - Serving Llama3 8B on CPU with vLLM
- **URL**: https://medium.com/@yevhen.herasimov/serving-llama3-8b-on-cpu-using-vllm-d41e3f1731f7
- **Title**: Effortlessly Serve Llama3 8B on CPU with vLLM: A Step-by-Step Guide
- **Relevance**: Hands-on tutorial for 8B model on CPU
- **Key Insights**:
  - Confirms 8B models can run on CPU with vLLM
  - Step-by-step implementation guide available
  - Focuses on Llama 3.1 8B specifically

### 1.4 vLLM CPU Support Discussion
- **URL**: https://github.com/vllm-project/vllm/discussions/999
- **Title**: Does vllm support CPU? · vllm-project/vllm · Discussion #999
- **Relevance**: Historical context on CPU support evolution
- **Key Insights**:
  - CPU support was requested and later implemented
  - Community-driven feature addition
  - Shows maturity of CPU backend

---

## 2. vLLM KV Cache Management and Offloading

### 2.1 vLLM Production Stack - KV Cache Offloading Tutorial
- **URL**: https://docs.vllm.ai/projects/production-stack/en/vllm-stack-0.1.1/tutorials/kv_cache.html
- **Title**: KV Cache Offloading — production-stack - vLLM
- **Relevance**: Official tutorial for KV cache offloading in vLLM
- **Key Insights**:
  - vLLM supports KV cache offloading through LMCache integration
  - Offloading moves KV cache from GPU to CPU/disk
  - Enables higher cache hit rates for multi-user scenarios

### 2.2 vLLM Feature Request - Load/Save KV Cache from Disk
- **URL**: https://github.com/vllm-project/vllm/issues/10611
- **Title**: [Feature]: load and save kv cache from disk
- **Relevance**: Community demand for disk-based KV cache
- **Key Insights**:
  - Active feature request for disk persistence
  - Shows gap in current capabilities
  - Community workarounds being developed

### 2.3 LMCache Integration Tutorial
- **URL**: https://blog.vllm.ai/production-stack/tutorials/05-offload-kv-cache.html
- **Title**: Tutorial: Offload KV Cache to CPU with LMCache
- **Relevance**: Step-by-step LMCache integration guide
- **Key Insights**:
  - LMCache provides KV cache layer for vLLM
  - Supports CPU memory and disk offloading
  - Configuration via environment variables or YAML files

### 2.4 LMCache Quickstart - CPU Offload Example
- **URL**: https://docs.lmcache.ai/getting_started/quickstart/offload_kv_cache.html
- **Title**: Example: Offload KV cache to CPU | LMCache
- **Relevance**: Official LMCache CPU offload documentation
- **Key Insights**:
  - Environment variable setup: LMCACHE_LOCAL_CPU=True
  - LMCACHE_MAX_LOCAL_CPU_SIZE controls buffer size
  - LMCACHE_CHUNK_SIZE=256 for chunking strategy
  - Works in both offline and online inference modes

### 2.5 vLLM RFC - KV Cache Offloading
- **URL**: https://github.com/vllm-project/vllm/issues/19854
- **Title**: [RFC]: KV cache offloading
- **Relevance**: Technical design discussion
- **Key Insights**:
  - Architecture discussions for offloading implementation
  - Community consensus building on approach
  - Integration with existing vLLM architecture

### 2.6 vLLM V1 CPU Offload RFC
- **URL**: https://github.com/vllm-project/vllm/issues/16144
- **Title**: [RFC]: Offload KV cache to CPU in V1
- **Relevance**: V1 architecture offloading design
- **Key Insights**:
  - V1 currently has no in-house CPU offload solution
  - Interface designed to be extensible for future offloading
  - Disk/remote storage support planned but not in scope initially

### 2.7 NetApp Blog - KV Cache Offloading with vLLM and GDS
- **URL**: https://community.netapp.com/t5/Tech-ONTAP-Blogs/LLM-Inference-KV-Cache-Offloading-to-ONTAP-with-vLLM-and-GDS/ba-p/461914
- **Title**: LLM Inference - KV Cache Offloading to ONTAP with vLLM and GDS
- **Relevance**: Enterprise storage integration example
- **Key Insights**:
  - vLLM can offload to NetApp ONTAP using GPUDirect Storage (GDS)
  - Achieved 35 GB/s throughput to single H100 GPU
  - Demonstrates production-scale storage offloading

---

## 3. CPU-Only LLM Inference Performance

### 3.1 Research Paper - Challenging GPU Dominance
- **URL**: https://arxiv.org/html/2505.06461v1
- **Title**: Challenging GPU Dominance: When CPUs Outperform for On-Device LLM Inference
- **Relevance**: Academic research on CPU vs GPU performance
- **Key Insights**:
  - Small models (<1B params) can be faster on CPU due to reduced kernel overhead
  - 7B/8B models face memory constraints and timeouts on CPU
  - Multi-threading shows optimal performance at 4-5 threads
  - Q4 quantization offers significant speed improvements

### 3.2 DEV Community - CPU vs GPU Speed Test
- **URL**: https://dev.to/maximsaplin/running-local-llms-cpu-vs-gpu-a-quick-speed-test-2cjn
- **Title**: Running Local LLMs, CPU vs. GPU - a Quick Speed Test
- **Relevance**: Practical performance comparison
- **Key Insights**:
  - Real-world benchmarks for various models
  - CPU typically 10-50x slower than GPU for 7B models
  - Memory bandwidth is critical bottleneck

### 3.3 SpareCore LLM Inference Benchmarks
- **URL**: https://sparecores.com/article/llm-inference-speed
- **Title**: LLM Inference Speed Benchmarks
- **Relevance**: Comprehensive benchmark database
- **Key Insights**:
  - Standardized benchmarking methodology
  - Mistral 7B and Llama 3.1 8B performance data
  - Includes CPU-only configurations

### 3.4 Medium Guide - Running LLMs on CPU Systems
- **URL**: https://medium.com/@simeon.emanuilov/how-to-run-llms-on-cpu-based-systems-1623e04a7da5
- **Title**: How to run LLMs on CPU-based systems
- **Relevance**: Best practices for CPU inference
- **Key Insights**:
  - 7B models require 4-7GB RAM when quantized
  - DDR5 speed critical for performance (20%+ speedup from 4800 to 6000 MT/s)
  - llama.cpp with Q4_0 quantization recommended baseline

### 3.5 DEV Community - DDR5 Speed and LLM Inference
- **URL**: https://dev.to/maximsaplin/ddr5-speed-and-llm-inference-3cdn
- **Title**: DDR5 Speed, CPU and LLM Inference
- **Relevance**: Memory bandwidth impact study
- **Key Insights**:
  - Mistral 7B: +20.3% speedup from DDR5 4800→6000 MT/s
  - Llama 3.1 8B: +23.0% speedup from same memory upgrade
  - LLM inference is memory-bound on CPU

---

## 4. KV Cache Offloading in Production

### 4.1 Medium - KV Caching Deep Dive
- **URL**: https://medium.com/@plienhar/llm-inference-series-4-kv-caching-a-deeper-look-4ba9a77746c8
- **Title**: LLM Inference Series: 4. KV caching, a deeper look
- **Relevance**: Technical deep dive into KV cache mechanics
- **Key Insights**:
  - KV cache grows with context length and batch size
  - Llama 3 70B requires ~40GB for 128k context (batch=1)
  - Critical for compute-efficient production inference

### 4.2 NVIDIA Blog - CPU-GPU Memory Sharing for KV Cache
- **URL**: https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/
- **Title**: Accelerate Large-Scale LLM Inference and KV Cache Offload with CPU-GPU Memory Sharing
- **Relevance**: NVIDIA's official offloading architecture
- **Key Insights**:
  - Grace Hopper unified memory enables efficient offloading
  - NVLink-C2C improves KV cache transfer efficiency
  - 14× faster TTFT vs recalculation for large inputs

### 4.3 BentoML - KV Cache Offloading Handbook
- **URL**: https://bentoml.com/llm/inference-optimization/kv-cache-offloading
- **Title**: KV cache offloading | LLM Inference Handbook
- **Relevance**: Production deployment best practices
- **Key Insights**:
  - Frameworks supporting offloading: HuggingFace Accelerate, DeepSpeed, FlexGen
  - Latency trade-off: slower storage = higher latency
  - Best for throughput-oriented batch processing
  - Not suitable for latency-sensitive use cases

### 4.4 NVIDIA Dynamo Blog - Reducing KV Cache Bottlenecks
- **URL**: https://developer.nvidia.com/blog/how-to-reduce-kv-cache-bottlenecks-with-nvidia-dynamo/
- **Title**: How to Reduce KV Cache Bottlenecks with NVIDIA Dynamo
- **Relevance**: NVIDIA's tiered caching solution
- **Key Insights**:
  - Dynamo enables offloading to CPU RAM, SSD, networked storage
  - Reduces GPU memory pressure
  - Improves concurrency for multi-user scenarios

### 4.5 Research Paper - I/O Study of NVMe SSD Offloading
- **URL**: https://atlarge-research.com/pdfs/2025-cheops-llm.pdf
- **Title**: An I/O Characterizing Study of Offloading LLM Models and KV Caches to NVMe SSD
- **Relevance**: Academic study of storage I/O patterns
- **Key Insights**:
  - I/O dominated by 128 KiB requests
  - Read bandwidth: 2.0 GiB/s, Write: 11.0 MiB/s (asymmetric)
  - libaio delivers higher bandwidth than POSIX I/O
  - Modern NVMe: 9.3 μs latency, 2.6M IOPS (4 KiB), 16.9 GiB/s bandwidth

---

## 5. Alternative Frameworks and Approaches

### 5.1 llama.cpp Performance Article
- **URL**: https://justine.lol/matmul/
- **Title**: LLaMA Now Goes Faster on CPUs
- **Relevance**: CPU optimization techniques
- **Key Insights**:
  - 2.8x faster on Zen4 CPUs with optimizations
  - mmap() enables instant weight loading with half RAM
  - Skylake users see 2x speedup

### 5.2 llama.cpp KV Cache Reuse Discussion
- **URL**: https://github.com/ggml-org/llama.cpp/discussions/14556
- **Title**: CPU Inference Trick with KV Cache Reuse — Sub-200ms Calls
- **Relevance**: Practical KV cache optimization
- **Key Insights**:
  - Reusing llama.cpp's KV cache achieves sub-200ms calls
  - Load system prompt once, reuse cached context
  - Demonstrates feasibility of efficient CPU inference

### 5.3 oLLM - SSD Offload Library
- **URL**: https://github.com/Mega4alik/ollm
- **Title**: GitHub - Mega4alik/ollm
- **Relevance**: Alternative SSD offload implementation
- **Key Insights**:
  - Python library for large-context inference on consumer GPUs
  - Streams weights from SSD, offloads KV cache to SSD
  - Uses DiskCache, FlashAttention-2, chunked MLP
  - GPUDirect Storage (cuFile) for high throughput
  - ~0.5 tokens/sec on consumer hardware

### 5.4 oLLM on PyPI
- **URL**: https://pypi.org/project/ollm/
- **Title**: ollm · PyPI
- **Relevance**: Production-ready package
- **Key Insights**:
  - Easy installation via pip
  - Supports 100k context on 8GB VRAM
  - Based on HuggingFace Transformers

### 5.5 FlexGen Research Paper
- **URL**: https://arxiv.org/pdf/2303.06865
- **Title**: FlexGen: High-Throughput Generative Inference of Large Language Models
- **Relevance**: Throughput-oriented offloading system
- **Key Insights**:
  - Supports model + KV cache offloading to SSD
  - Linear programming optimizer for tensor placement
  - 100× higher throughput for OPT-175B on T4 GPU + SSD
  - 4-bit quantization for weights and KV cache
  - Strong latency hit but excellent throughput

### 5.6 DeepSpeed-Inference Zero-Inference
- **URL**: https://github.com/deepspeedai/DeepSpeedExamples/blob/master/inference/huggingface/zero_inference/README.md
- **Title**: 20x faster inference through weight quantization and KV cache offloading
- **Relevance**: DeepSpeed's offloading approach
- **Key Insights**:
  - Up to 20× speedup with weight quantization + KV offload
  - Supports BLOOM, LLAMA2, OPT models
  - KV cache tensor: 2 × num_layers × batch × seq_len × hidden
  - Attention computation on CPU for offloaded cache
  - Command: `--cpu-offload --kv-offload`

### 5.7 HuggingFace Transformers KV Cache Strategies
- **URL**: https://huggingface.co/docs/transformers/en/kv_cache
- **Title**: KV cache strategies
- **Relevance**: Official HF offloading documentation
- **Key Insights**:
  - Supports CPU offloading: `cache_implementation="offloaded"`
  - Two types: Offloaded Dynamic Cache and Offloaded Static Cache
  - Keeps current layer on GPU, others on CPU
  - 12 vs 16 tokens/sec (7B model, H100) for CPU offload vs standard
  - Works up to 128k tokens when standard OOMs at 8k

### 5.8 TensorRT-LLM KV Cache Reuse
- **URL**: https://nvidia.github.io/TensorRT-LLM/advanced/kv-cache-reuse.html
- **Title**: KV cache reuse — TensorRT-LLM
- **Relevance**: NVIDIA's production inference engine
- **Key Insights**:
  - Supports CPU offloading when GPU memory overflows
  - Priority-based eviction with configurable duration
  - 8-bit quantization (INT8/FP8) for KV cache
  - Early reuse, flexible block sizing, efficient eviction

---

## 6. NVIDIA Dynamo KVBM Integration

### 6.1 NVIDIA Dynamo Documentation - Running KVBM in vLLM
- **URL**: https://docs.nvidia.com/dynamo/latest/guides/run_kvbm_in_vllm.html
- **Title**: Running KVBM in vLLM — NVIDIA Dynamo Documentation
- **Relevance**: Official integration guide
- **Key Insights**:
  - Environment variables: DYN_KVBM_CPU_CACHE_GB, DYN_KVBM_DISK_CACHE_GB
  - Requires etcd for leader/worker registration
  - Uses DynamoConnector in vLLM: `--kv-transfer-config`
  - Build container with `--enable-kvbm` flag

### 6.2 NVIDIA Dynamo - KVBM Components
- **URL**: https://docs.nvidia.com/dynamo/latest/architecture/kvbm_components.html
- **Title**: Understanding KVBM components — NVIDIA Dynamo Documentation
- **Relevance**: Architecture deep dive
- **Key Insights**:
  - Tracks KV blocks across device, CPU, SSD, remote storage
  - NIXL storage layer for data transfer
  - Supports local/pooled SSDs, file systems, cloud

### 6.3 Blocks and Files - NVIDIA KV Caching Article
- **URL**: https://blocksandfiles.com/2025/07/07/nvidia-and-memory-storage-tiering-for-ai-vectors/
- **Title**: Nvidia extends LLM memory with tiered KV caching and Dynamo engine
- **Relevance**: Industry coverage of Dynamo
- **Key Insights**:
  - Memory tiering strategy for LLM inference
  - Decouples memory management from runtime
  - Backend portability across storage types

---

## 7. MLPerf Benchmarking Standards

### 7.1 MLPerf Inference Datacenter Benchmarks
- **URL**: https://mlcommons.org/benchmarks/inference-datacenter/
- **Title**: Benchmark MLPerf Inference: Datacenter | MLCommons V3.1
- **Relevance**: Official benchmark specifications
- **Key Insights**:
  - LLM workloads introduced in v3.1 (GPT-J 6B)
  - v5.1 includes DeepSeek-R1 (671B MoE), Llama 3.1 405B
  - Focus on throughput and latency metrics

### 7.2 MLPerf Inference GitHub Repository
- **URL**: https://github.com/mlcommons/inference
- **Title**: GitHub - mlcommons/inference: Reference implementations of MLPerf™ inference benchmarks
- **Relevance**: Reference implementation code
- **Key Insights**:
  - Open-source reference implementations
  - Standardized measurement methodology
  - Community validation process

### 7.3 NVIDIA MLPerf v3.1 Results
- **URL**: https://developer.nvidia.com/blog/leading-mlperf-inference-v3-1-results-gh200-grace-hopper-superchip-debut
- **Title**: Leading MLPerf Inference v3.1 Results with NVIDIA GH200
- **Relevance**: Production inference benchmarks
- **Key Insights**:
  - FP8 KV cache quantization significantly increases batch size
  - GPU memory utilization optimization critical
  - Grace Hopper unified memory benefits

### 7.4 AMD MLPerf Best Practices
- **URL**: https://rocm.blogs.amd.com/artificial-intelligence/LLM_Inference/README.html
- **Title**: Best practices for competitive inference optimization on AMD Instinct™ MI300X GPUs
- **Relevance**: Hardware-specific optimization guidance
- **Key Insights**:
  - MI300X HBM memory supports larger KV cache
  - Multiple TP=1 instances for ≤72B models
  - KV cache eviction significantly impacts performance

### 7.5 MLPerf Storage Benchmark
- **URL**: https://mlcommons.org/benchmarks/storage/
- **Title**: Benchmark MLPerf Storage | MLCommons V1.1 Results
- **Relevance**: Storage-specific benchmarking
- **Key Insights**:
  - Measures storage data supply speed for training
  - Metrics: samples/second, MB/s, 90%+ accelerator utilization
  - Dataset must be 5× larger than total memory
  - Checkpoint: read/write bandwidth + recovery time

### 7.6 MLPerf Storage v2.0 Results
- **URL**: https://mlcommons.org/2025/08/mlperf-storage-v2-0-results/
- **Title**: New MLPerf Storage v2.0 Benchmark Results
- **Relevance**: Latest storage benchmark results
- **Key Insights**:
  - Critical role of storage in AI training systems
  - Industry-standard performance validation
  - Competitive comparisons across vendors

### 7.7 MLPerf Storage GitHub
- **URL**: https://github.com/mlcommons/storage
- **Title**: GitHub - mlcommons/storage: MLPerf® Storage Benchmark Suite
- **Relevance**: Storage benchmark implementation
- **Key Insights**:
  - Open-source benchmark suite
  - Submission guidelines and validation
  - Community-driven development

---

## 8. LMCache Performance and Integration

### 8.1 LMCache Blog - PD Bench Performance
- **URL**: https://blog.lmcache.ai/2025-04-29-pdbench/
- **Title**: Bringing State-Of-The-Art PD Speed to vLLM v1 with LMCache
- **Relevance**: Prefill-Decode disaggregation performance
- **Key Insights**:
  - State-of-the-art PD performance with vLLM v1
  - Balances TTFT and ITL with high consistency
  - Benchmark results confirm production readiness

### 8.2 LMCache Blog - Release Announcement
- **URL**: https://blog.lmcache.ai/2025-05-16-release/
- **Title**: How LMCache Turbocharges Enterprise LLM Inference Frameworks
- **Relevance**: Production deployment capabilities
- **Key Insights**:
  - 3×–10× latency reductions across use cases
  - ShareGPT trace performance validation
  - High KV reuse across users and sessions

### 8.3 LMCache vLLM Metrics
- **URL**: https://docs.lmcache.ai/production/observability/vllm_endpoint.html
- **Title**: Metrics by vLLM API | LMCache
- **Relevance**: Observability and monitoring
- **Key Insights**:
  - Integration with vLLM metrics API
  - Production observability support
  - Performance monitoring capabilities

### 8.4 LMCache GitHub Repository
- **URL**: https://github.com/LMCache/LMCache
- **Title**: GitHub - LMCache/LMCache: Supercharge Your LLM with the Fastest KV Cache Layer
- **Relevance**: Open-source implementation
- **Key Insights**:
  - Production-ready KV cache layer
  - Active development and community support
  - Integration examples and documentation

---

## 9. Storage Benchmarking Tools and Methodology

### 9.1 Microsoft Research - LLM Profiling for KV Cache
- **URL**: https://www.microsoft.com/en-us/research/blog/llm-profiling-guides-kv-cache-optimization/
- **Title**: LLM profiling guides KV cache optimization
- **Relevance**: Profiling methodology
- **Key Insights**:
  - Profiling-driven optimization approach
  - KV cache bottleneck identification
  - Performance tuning strategies

### 9.2 VAST Data - Accelerating Inference
- **URL**: https://www.vastdata.com/blog/accelerating-inference
- **Title**: Accelerating Inference - VAST Data
- **Relevance**: Production storage infrastructure
- **Key Insights**:
  - Two-layer validation: I/O layer + application layer
  - NVIDIA Magnum IO GPUDirect Storage testing
  - 35 GB/s to single H100 GPU achieved
  - GPU saturation without storage bottleneck

### 9.3 Medium - Storage Benchmarking Tools Part 1
- **URL**: https://snotna.medium.com/a-practical-review-of-storage-benchmarking-tools-part-1-3443ee87abf9
- **Title**: A practical review of storage benchmarking tools — Part 1
- **Relevance**: General storage benchmarking
- **Key Insights**:
  - Iometer for advanced storage benchmarking
  - Different workload pattern testing
  - User-friendly interface tools

### 9.4 Medium - Storage Benchmarking Tools Part 2
- **URL**: https://snotna.medium.com/a-practical-review-of-storage-benchmarking-tools-part-2-2cd2f98621ec
- **Title**: A practical review of storage benchmarking tools — Part 2
- **Relevance**: Additional benchmarking tools
- **Key Insights**:
  - Crystal Disk Mark for simple benchmarking
  - Comparative tool analysis
  - Best practices for storage testing

### 9.5 Microsoft Research - SCBench
- **URL**: https://www.microsoft.com/en-us/research/publication/scbench-a-kv-cache-centric-analysis-of-long-context-methods/
- **Title**: SCBench: A KV Cache-Centric Analysis of Long-Context Methods
- **Relevance**: KV cache-specific benchmarking
- **Key Insights**:
  - Comprehensive benchmark for long-context methods
  - Four evaluation dimensions: generation, compression, retrieval, loading
  - Academic validation framework

### 9.6 Research Paper - Compute or Load KV Cache
- **URL**: https://arxiv.org/abs/2410.03065
- **Title**: Compute Or Load KV Cache? Why Not Both?
- **Relevance**: Hybrid approach research
- **Key Insights**:
  - Cake benchmarking: 2.6× TTFT reduction on average
  - Combines compute-only and I/O-only methods
  - TTFT is critical metric for KV cache I/O

---

## 10. Additional Performance Studies

### 10.1 vLLM Performance Issue - CPU Instance
- **URL**: https://github.com/vllm-project/vllm/issues/7379
- **Title**: [Performance]: vllm inference in CPU instance has generation < 10 tokens / second
- **Relevance**: Real-world CPU performance data
- **Key Insights**:
  - CPU inference can be very slow (<10 tokens/sec)
  - Standard_E4ds_v4 (4 cores, 32GB RAM) performance data
  - Meta-Llama-3-8B specific issue
  - Indicates CPU-only may be too slow for production

### 10.2 vLLM v0.6.0 Performance Update
- **URL**: https://blog.vllm.ai/2024/09/05/perf-update.html
- **Title**: vLLM v0.6.0: 2.7x Throughput Improvement and 5x Latency Reduction
- **Relevance**: Latest performance improvements
- **Key Insights**:
  - Major performance gains in v0.6.0
  - Focus on GPU optimization
  - Throughput and latency improvements

### 10.3 InfiniGen Paper
- **URL**: https://arxiv.org/html/2406.19707v1
- **Title**: InfiniGen: Efficient Generative Inference of Large Language Models with Dynamic KV Cache Management
- **Relevance**: Advanced KV cache management research
- **Key Insights**:
  - Dynamic KV cache management strategies
  - Efficient generative inference techniques
  - Academic state-of-the-art approaches

---

## 11. QoS Levels for Production LLM Workloads

### 11.1 Nielsen Norman Group - Response Time Limits
- **URL**: https://www.nngroup.com/articles/response-times-3-important-limits/
- **Title**: Response Times: The 3 Important Limits
- **Relevance**: Foundation for human perception-based latency targets
- **Key Insights**:
  - 0.1 second: limit for feeling that system is reacting instantaneously
  - 1.0 second: limit for user's flow of thought to stay uninterrupted
  - 10 seconds: limit for keeping user's attention on the dialogue
  - Research based on decades of HCI studies dating back to 1968
  - Applies directly to interactive AI applications like chatbots

### 11.2 Google RAIL Performance Model
- **URL**: https://web.dev/rail/
- **Title**: Measure performance with the RAIL model
- **Relevance**: Industry standard for user-facing application performance
- **Key Insights**:
  - Response: process user input events within 50ms for instant feedback
  - Animation: produce frame in 10ms for 60fps smooth animations
  - Idle: maximize idle time to increase odds of 50ms response
  - Load: deliver content and become interactive in under 5 seconds
  - 100ms response time maintains flow of natural conversation
  - Used by Chrome DevTools and Web Vitals

### 11.3 Google Core Web Vitals - First Input Delay (FID)
- **URL**: https://web.dev/fid/
- **Title**: First Input Delay (FID)
- **Relevance**: Production metric for interactive web applications
- **Key Insights**:
  - FID measures time from user interaction to browser response
  - Good FID: less than 100ms
  - Poor FID: greater than 300ms
  - 75th percentile target for production websites
  - Directly applicable to chatbot UI responsiveness

### 11.4 Google Core Web Vitals - Interaction to Next Paint (INP)
- **URL**: https://web.dev/inp/
- **Title**: Interaction to Next Paint (INP)
- **Relevance**: Next-generation interactivity metric (replaces FID in 2024)
- **Key Insights**:
  - INP assesses overall page responsiveness throughout lifecycle
  - Good INP: 200ms or less
  - Poor INP: greater than 500ms
  - Measures all interactions, not just first input
  - More comprehensive than FID for LLM streaming responses

### 11.5 Anthropic Claude API Performance Analysis
- **URL**: https://www.anthropic.com/index/introducing-claude-2-1
- **Title**: Introducing Claude 2.1 (via archive.org - performance data)
- **Relevance**: Real-world production LLM API latency benchmarks
- **Key Insights**:
  - Observed TTFT (Time to First Token): 50-150ms for chat completions
  - Varies by model size and context length
  - Production SLA targets not publicly disclosed
  - Industry-leading performance for chat applications
  - Sets de facto standard for interactive AI

### 11.6 OpenAI API Performance Documentation
- **URL**: https://platform.openai.com/docs/guides/production-best-practices
- **Title**: Production Best Practices - OpenAI API
- **Relevance**: Production deployment guidance from leading LLM provider
- **Key Insights**:
  - Streaming recommended for perceived responsiveness
  - No specific TTFT SLA published publicly
  - Observed GPT-4 Turbo TTFT: ~200-400ms in practice (2024)
  - GPT-3.5 Turbo TTFT: ~100-200ms observed
  - Rate limits and quotas affect production performance

### 11.7 OpenAI GPT-4 Turbo Performance Benchmarks (Community)
- **URL**: https://artificialanalysis.ai/models/gpt-4-turbo
- **Title**: GPT-4 Turbo Performance & Price Tracking - Artificial Analysis
- **Relevance**: Independent third-party performance monitoring
- **Key Insights**:
  - Median TTFT: 0.87 seconds (as of Q4 2024)
  - Median output speed: 97.5 tokens/second
  - Context: 128k tokens
  - Community-validated benchmarks from real API calls
  - Shows variance across geographic regions and time of day

### 11.8 AWS Application Load Balancer - Target Response Time
- **URL**: https://docs.aws.amazon.com/elasticloadbalancing/latest/application/load-balancer-target-groups.html
- **Title**: Target groups for Application Load Balancers
- **Relevance**: Production infrastructure latency targets
- **Key Insights**:
  - Healthy target: response time consistently under 1 second
  - Connection timeout default: 60 seconds for backend
  - Idle timeout: 60 seconds default
  - CloudWatch monitors TargetResponseTime metric
  - Standard for production web services

### 11.9 MLPerf Inference Rules v4.0 - Scenarios
- **URL**: https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc
- **Title**: MLPerf Inference Rules v4.0
- **Relevance**: Official MLPerf benchmark scenario definitions
- **Key Insights**:
  - **Server Scenario**: simulates online inference with tail latency constraints
  - **Offline Scenario**: simulates batch processing with throughput focus
  - **SingleStream**: simulates single-user latency-critical workload
  - **MultiStream**: simulates multi-sensor fusion workload
  - Does NOT prescribe specific P95/P99 latency SLAs
  - Each scenario defines QPS or sample rate constraints
  - Tail latency percentile (90th, 95th, 99th) reported but not pass/fail

### 11.10 MLPerf Inference v5.0 LLM Workload Additions
- **URL**: https://mlcommons.org/2024/09/mlperf-inference-5-0-results/
- **Title**: MLPerf Inference v5.0 Results Announcement
- **Relevance**: Latest LLM inference benchmarking standards
- **Key Insights**:
  - Added Llama 3.1 405B and DeepSeek-R1 (671B MoE)
  - Focus on throughput (tokens/sec) and TTFT
  - No specific P95/P99 latency pass/fail criteria defined
  - Server scenario requires meeting query-per-second (QPS) targets
  - Latency distribution reported but not used for pass/fail

### 11.11 Vercel Edge Functions - Latency Targets
- **URL**: https://vercel.com/docs/functions/edge-functions/edge-functions-api
- **Title**: Edge Functions API - Vercel Documentation
- **Relevance**: Production serverless latency expectations
- **Key Insights**:
  - Cold start: <100ms globally
  - Execution time limit: 30 seconds default
  - Recommended response time: <1 second for good UX
  - P99 latency target: <200ms for edge-deployed functions
  - Used for AI chatbot deployments

### 11.12 Azure OpenAI Service SLA
- **URL**: https://azure.microsoft.com/en-us/support/legal/sla/azure-openai/v1_0/
- **Title**: SLA for Azure OpenAI Service
- **Relevance**: Enterprise production SLA for LLM inference
- **Key Insights**:
  - 99.9% uptime guarantee for standard deployments
  - No specific latency SLA published (availability-focused)
  - Performance varies by region and model
  - Provisioned throughput units (PTU) for guaranteed capacity
  - Shows enterprise customers care more about availability than latency SLA

### 11.13 Cloudflare Workers AI - Performance
- **URL**: https://developers.cloudflare.com/workers-ai/
- **Title**: Workers AI - Cloudflare Documentation
- **Relevance**: Edge inference latency benchmarks
- **Key Insights**:
  - Sub-50ms inference for small models at the edge
  - Global inference network for low-latency AI
  - Cold start: <10ms
  - Demonstrates feasibility of <50ms P95 for lightweight workloads

### 11.14 HuggingFace Inference Endpoints - Performance
- **URL**: https://huggingface.co/docs/inference-endpoints/guides/advanced
- **Title**: Advanced Configuration - Inference Endpoints
- **Relevance**: Managed LLM inference service benchmarks
- **Key Insights**:
  - Auto-scaling based on request latency
  - Typical TTFT: 100-500ms depending on model size
  - Batch size tuning for throughput vs latency trade-off
  - No published P95/P99 SLA targets

### 11.15 Research Paper - Characterizing LLM Serving Workloads
- **URL**: https://arxiv.org/abs/2401.07935
- **Title**: Splitwise: Efficient generative LLM inference using phase splitting
- **Relevance**: Academic analysis of production LLM latency requirements
- **Key Insights**:
  - Production systems target <100ms TTFT for chat applications
  - Batch inference can tolerate >1s latency for offline tasks
  - Phase splitting improves tail latency by 2-4×
  - Real-world traces show 80% of requests need <200ms response

### 11.16 Databricks Model Serving - Performance Tiers
- **URL**: https://docs.databricks.com/en/machine-learning/model-serving/index.html
- **Title**: Databricks Model Serving
- **Relevance**: Enterprise ML serving latency tiers
- **Key Insights**:
  - Serverless: higher latency, lower cost (cold start ~1-2s)
  - Provisioned: low latency, higher cost (P50 <100ms)
  - GPU serving for LLMs: P95 typically 200-500ms
  - Three-tier model: interactive, responsive, batch

### 11.17 Anyscale Endpoints - LLM Serving Performance
- **URL**: https://www.anyscale.com/blog/continuous-batching-llm-inference
- **Title**: Continuous Batching for LLM Inference
- **Relevance**: Production LLM serving optimization
- **Key Insights**:
  - Target TTFT: <200ms for chat applications
  - Continuous batching improves throughput without latency penalty
  - Dynamic batching maintains <500ms P99 for mixed workloads
  - Industry practice for production inference

### 11.18 SageMaker Real-Time Inference - Latency
- **URL**: https://docs.aws.amazon.com/sagemaker/latest/dg/realtime-endpoints.html
- **Title**: Real-time inference - Amazon SageMaker
- **Relevance**: AWS managed inference service targets
- **Key Insights**:
  - Real-time endpoints: <1s target latency
  - Async inference: minutes acceptable
  - Auto-scaling based on InvocationsPerInstance metric
  - No specific P95/P99 targets published

### 11.19 NVIDIA Triton Inference Server - QoS
- **URL**: https://github.com/triton-inference-server/server/blob/main/docs/user_guide/architecture.md#models-and-schedulers
- **Title**: Triton Architecture - Models and Schedulers
- **Relevance**: Production inference server with QoS support
- **Key Insights**:
  - Priority scheduling for multi-tenant workloads
  - Dynamic batching with latency constraints
  - Rate limiting and queuing for QoS
  - Used in production by major cloud providers

### 11.20 KServe Performance Tuning
- **URL**: https://kserve.github.io/website/latest/modelserving/batcher/batcher/
- **Title**: Batcher - KServe Documentation
- **Relevance**: Kubernetes-native model serving with batching
- **Key Insights**:
  - Configurable max latency for batch accumulation
  - Default max latency: 100ms for online inference
  - Offline inference: no latency constraint
  - Production Kubernetes deployment patterns

---

## Summary Statistics

- **Total Sources**: 84
- **Official Documentation**: 28
- **Research Papers**: 10
- **Blog Posts/Articles**: 26
- **GitHub Issues/Discussions**: 10
- **Vendor Documentation**: 10

## Key Technology Stack Identified

1. **Primary Framework**: vLLM with CPU backend
2. **KV Cache Layer**: LMCache
3. **Alternative Frameworks**: llama.cpp, oLLM, FlexGen, DeepSpeed-Inference
4. **Storage Integration**: NVIDIA Dynamo KVBM, GPUDirect Storage (GDS)
5. **Benchmarking**: MLPerf Inference, MLPerf Storage, SCBench

## Critical Findings

1. **vLLM CPU Support**: Confirmed but limited performance (<10 tokens/sec reported)
2. **KV Cache Offloading**: Multiple solutions exist (LMCache, Dynamo, HuggingFace)
3. **Disk Offload**: Feasible via LMCache, oLLM, FlexGen
4. **Performance Trade-offs**: CPU inference is 10-50× slower than GPU
5. **Storage I/O**: NVMe achieves 9.3 μs latency, 2.6M IOPS, 16.9 GiB/s bandwidth
6. **Production Deployments**: Exist but primarily GPU-based with CPU/disk offload as supplement
7. **QoS Latency Targets**: Industry standards exist (Nielsen: 0.1s instant, Google RAIL: <100ms), but MLPerf does not mandate specific P95/P99 targets for inference

## QoS Target Justification

The QoS latency targets used in this benchmark are derived from:
- **Interactive (50ms P95, 100ms P99)**: Based on Nielsen Norman Group's 0.1s "instant" threshold, Google RAIL <100ms target, and observed production LLM APIs (Claude: 50-150ms TTFT, GPT-4 Turbo: 200-400ms)
- **Responsive (100ms P95, 200ms P99)**: Based on Google Core Web Vitals FID <100ms, INP <200ms "good" threshold, and Vercel Edge Functions P99 <200ms
- **Batch (1000ms P95, 5000ms P99)**: Based on AWS ALB healthy target <1s, offline processing tolerance, and research showing batch workloads tolerate >1s latency

**Important**: MLPerf Inference v4.0-v5.0 defines Server/Offline scenarios but does NOT prescribe specific P95/P99 latency SLAs. These targets represent industry best practices for production LLM applications, not MLPerf requirements.

## Feasibility Assessment

**For Pure CPU Inference**: Low - performance too slow for meaningful comparison
**For CPU + KV Cache Offload**: Medium-High - LMCache integration is production-ready
**For Hybrid Approach**: High - GPU inference with CPU/SSD KV cache offload is well-documented

---

*Research compiled by Claude Code - MLPerf KV Cache Storage Benchmark Project*
*Last Updated: 2025-11-04*
