#!/usr/bin/env python3
"""
test_multi_endpoint_s3dlio.py
------------------------------
Demonstrates s3dlio's native MultiEndpointStore (round-robin load balancing
across multiple S3 servers) without any mlpstorage/DLIO overhead.

Two s3-ultra servers must be running:
  - http://127.0.0.1:9101  (bucket: mlp-s3dlio)
  - http://127.0.0.1:9102  (bucket: mlp-s3dlio)

Run from the mlp-storage environment:
  uv run python tests/object-store/test_multi_endpoint_s3dlio.py
"""

import asyncio
import os
import sys
import time

# Credentials for the local s3-ultra servers
os.environ["AWS_ACCESS_KEY_ID"] = "testkey"
os.environ["AWS_SECRET_ACCESS_KEY"] = "testsecret"

import s3dlio  # noqa: E402  (env vars must be set before import)


EP1 = "http://127.0.0.1:9101"
EP2 = "http://127.0.0.1:9102"
BUCKET = "mlp-s3dlio"
PREFIX = "multi-ep-test"
NUM_OBJECTS = 200          # total objects to PUT
OBJECT_SIZE = 32 * 1024    # 32 KiB each  (~6.25 MiB total — fast test)
CONCURRENCY = 32           # asyncio.gather batch size


def _make_root_uri(endpoint_url: str, bucket: str) -> str:
    """Convert http://host:port to s3://host:port/bucket/"""
    host_port = endpoint_url.replace("http://", "").replace("https://", "")
    return f"s3://{host_port}/{bucket}/"


async def run_test() -> None:
    ep1_root = _make_root_uri(EP1, BUCKET)
    ep2_root = _make_root_uri(EP2, BUCKET)

    print(f"\n{'='*60}")
    print("s3dlio Native MultiEndpointStore Test")
    print(f"{'='*60}")
    print(f"Endpoint 1 : {EP1}  (root: {ep1_root})")
    print(f"Endpoint 2 : {EP2}  (root: {ep2_root})")
    print(f"Objects    : {NUM_OBJECTS}  ({OBJECT_SIZE // 1024} KiB each)")
    print(f"Strategy   : round_robin")
    print(f"{'='*60}\n")

    store = s3dlio.create_multi_endpoint_store(
        uris=[ep1_root, ep2_root],
        strategy="round_robin",
    )
    print(f"Store created: {store.endpoint_count} endpoints, strategy={store.strategy}")

    # Generate deterministic test payload
    payload = bytes(range(256)) * (OBJECT_SIZE // 256)

    # PUT all objects concurrently in batches
    print(f"\nPUT {NUM_OBJECTS} objects in batches of {CONCURRENCY}...")
    t0 = time.perf_counter()
    put_errors = 0

    for batch_start in range(0, NUM_OBJECTS, CONCURRENCY):
        batch = range(batch_start, min(batch_start + CONCURRENCY, NUM_OBJECTS))
        tasks = [
            store.put(f"{PREFIX}/obj_{i:06d}.bin", payload)
            for i in batch
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for idx, r in zip(batch, results):
            if isinstance(r, Exception):
                print(f"  ERROR obj_{idx:06d}: {r}", file=sys.stderr)
                put_errors += 1

    elapsed = time.perf_counter() - t0
    total_bytes = NUM_OBJECTS * OBJECT_SIZE
    throughput = total_bytes / elapsed / 1024 / 1024

    print(f"PUT complete: {NUM_OBJECTS - put_errors}/{NUM_OBJECTS} succeeded")
    print(f"  Elapsed : {elapsed:.2f}s")
    print(f"  Throughput: {throughput:.1f} MiB/s")

    # --- Per-endpoint stats ---
    print(f"\n{'='*60}")
    print("Per-Endpoint Statistics (after PUTs)")
    print(f"{'='*60}")
    stats = store.get_endpoint_stats()
    for ep_stat in stats:
        uri = ep_stat["uri"]
        reqs = ep_stat["total_requests"]
        written_kb = ep_stat["bytes_written"] / 1024
        errors = ep_stat["error_count"]
        print(f"  {uri}")
        print(f"    requests : {reqs}")
        print(f"    written  : {written_kb:.1f} KiB  ({ep_stat['bytes_written']:,} bytes)")
        print(f"    errors   : {errors}")

    total_stats = store.get_total_stats()
    print(f"\nTotal across all endpoints:")
    print(f"  requests : {total_stats['total_requests']}")
    print(f"  written  : {total_stats['bytes_written'] / 1024:.1f} KiB")

    # Expect roughly equal distribution (round-robin)
    if len(stats) == 2:
        r0 = stats[0]["total_requests"]
        r1 = stats[1]["total_requests"]
        balance = min(r0, r1) / max(r0, r1) if max(r0, r1) > 0 else 0.0
        print(f"\nLoad balance ratio: {r0}:{r1}  ({balance*100:.1f}% balanced)")
        if balance >= 0.8:
            print("PASS: Both endpoints received data (>80% balanced)")
        else:
            print("WARN: Load distribution is uneven (< 80% balanced)")

    # The per-endpoint stats from the MultiEndpointStore ARE the authoritative
    # distribution proof: they record exactly how many bytes/requests each endpoint
    # received during this store's lifetime.  s3dlio caches per-endpoint stores
    # internally, so trying to use s3dlio.list() with a changed AWS_ENDPOINT_URL
    # after the multi-endpoint store is created is unreliable.  Stats suffice.
    ep1_reqs = stats[0]["total_requests"] if len(stats) > 0 else 0
    ep2_reqs = stats[1]["total_requests"] if len(stats) > 1 else 0
    verify_ok = (ep1_reqs + ep2_reqs == NUM_OBJECTS) and ep1_reqs > 0 and ep2_reqs > 0

    # Cleanup
    print(f"\nCleaning up {NUM_OBJECTS} distributed test objects...")
    del_tasks = [store.delete(f"{PREFIX}/obj_{i:06d}.bin") for i in range(NUM_OBJECTS)]
    del_results = await asyncio.gather(*del_tasks, return_exceptions=True)
    del_errors = sum(1 for r in del_results if isinstance(r, Exception))
    print(f"Deleted {NUM_OBJECTS - del_errors}/{NUM_OBJECTS} objects")

    print(f"\n{'='*60}")
    if put_errors == 0 and verify_ok:
        print("RESULT: PASS — s3dlio native multi-endpoint PUT distribution works")
    else:
        print(f"RESULT: FAIL — {put_errors} PUT errors, distribution check: {verify_ok}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(run_test())
