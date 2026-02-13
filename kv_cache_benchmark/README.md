# MLPerf Storage KV Cache Benchmark

This directory contains the initial implementation of the KV Cache benchmark for MLPerf Storage v3.

## Overview

The KV Cache benchmark simulates the storage access patterns of Large Language Model (LLM) inference systems, specifically focusing on key-value cache operations that are critical for multi-turn conversations and long-context processing.

## Components

### Core Scripts

- **kv-cache.py**: Main benchmark implementation for KV cache storage performance testing
- **kv-cache_sharegpt_replay.py**: ShareGPT conversation replay-based benchmark for realistic workload simulation
- **kv-cache-wrapper.sh**: Wrapper script for running benchmark configurations
- **validate.sh**: Validation script for benchmark results

### Documentation

- **MLperf v3 KV cache proposal.md**: Detailed proposal for KV cache benchmark integration into MLPerf Storage
- **MLperf v3 KV cache proposal.pdf**: PDF version of the proposal
- **sources.md**: References and source documentation

## Purpose

This benchmark addresses the growing need to measure storage system performance under AI/ML inference workloads, particularly:

- Key-value cache read/write patterns
- Mixed sequential and random access patterns
- Multi-threaded concurrent access
- Realistic conversation-based workload replay

## Getting Started

See the proposal documents for detailed information about the benchmark design, metrics, and validation criteria.

## Status

Initial implementation - work in progress for MLPerf Storage v3.0
