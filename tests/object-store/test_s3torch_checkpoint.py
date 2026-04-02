#!/usr/bin/env python3
"""S3TorchConnector streaming checkpoint test.

Credential precedence: .env file < environment variables < CLI options
"""

import os
import sys
import time
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def load_env_config():
    env_path = None
    for candidate in [
        Path(__file__).parent.parent / ".env",
        Path(__file__).parent / ".env",
        Path.cwd() / ".env",
    ]:
        if candidate.exists():
            env_path = candidate
            break

    config = {}
    if env_path:
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, _, val = line.partition('=')
                    config[key.strip()] = val.strip()
        print(f"Loaded credentials from: {env_path}")
    else:
        print("No .env file found, using environment variables")

    for key in ['AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'AWS_ENDPOINT_URL', 'AWS_REGION']:
        if key in os.environ:
            config[key] = os.environ[key]

    return config


def apply_config(config: dict):
    for key, val in config.items():
        os.environ[key] = val



def test_s3torch_checkpoint(uri: str, size_gb: float):
    from mlpstorage.checkpointing import StreamingCheckpointing

    total_bytes = int(size_gb * (1024**3))

    print("=" * 80)
    print("S3TORCHCONNECTOR CHECKPOINT TEST")
    print("=" * 80)
    print(f"URI:       {uri}")
    print(f"Size:      {size_gb:.2f} GB")
    print(f"Multipart: Auto-managed by s3torchconnector")
    print("=" * 80)
    print()

    checkpoint = StreamingCheckpointing(
        chunk_size=32 * 1024 * 1024,
        num_buffers=4,
        use_dgen=True,
        backend='s3torchconnector',
    )

    try:
        start = time.perf_counter()
        result = checkpoint.save(uri, total_bytes)
        elapsed = time.perf_counter() - start
        io_throughput = result.get('io_throughput_gbps', size_gb / elapsed)

        print()
        print("=" * 80)
        print("✅ SUCCESS")
        print("=" * 80)
        print(f"Time:             {elapsed:.2f}s")
        print(f"I/O Throughput:   {io_throughput:.2f} GB/s")
        print(f"Total Throughput: {size_gb / elapsed:.2f} GB/s")
        if 'memory_usage_mb' in result:
            print(f"Memory:           {result['memory_usage_mb']:.1f} MB")
        print("=" * 80)
        return True
    except Exception as e:
        print()
        print("=" * 80)
        print(f"❌ FAILED: {e}")
        print("=" * 80)
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description='S3TorchConnector streaming checkpoint test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--bucket', default='bucket-s3torch', help='S3 bucket name')
    parser.add_argument('--key', default=None,
                        help='Object key (default: auto-generated with timestamp)')
    parser.add_argument('--s3-uri', default=None,
                        help='Full S3 URI (overrides --bucket / --key)')
    parser.add_argument('--size-gb', type=float, default=1.0, help='Checkpoint size in GB')
    parser.add_argument('--endpoint', default=None, help='S3 endpoint URL')
    parser.add_argument('--access-key', default=None, help='AWS/MinIO access key')
    parser.add_argument('--secret-key', default=None, help='AWS/MinIO secret key')
    parser.add_argument('--region', default=None, help='AWS region')
    args = parser.parse_args()

    config = load_env_config()
    if args.endpoint:
        config['AWS_ENDPOINT_URL'] = args.endpoint
    if args.access_key:
        config['AWS_ACCESS_KEY_ID'] = args.access_key
    if args.secret_key:
        config['AWS_SECRET_ACCESS_KEY'] = args.secret_key
    if args.region:
        config['AWS_REGION'] = args.region
    apply_config(config)

    if args.s3_uri:
        uri = args.s3_uri
    else:
        key = args.key or f"test/s3torch-checkpoint-{int(time.time())}.dat"
        uri = f"s3://{args.bucket}/{key}"

    success = test_s3torch_checkpoint(uri, args.size_gb)
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
