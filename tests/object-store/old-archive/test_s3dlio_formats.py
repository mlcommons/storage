#!/usr/bin/env python3
"""
test_s3dlio_formats.py
======================

Standalone (no pytest) integration test: put + get for every DLIO data format
via DLIOBenchmark + s3dlio against real S3-compatible object storage (MinIO).

Each format runs a 3-phase cycle:
  Phase 1 — generate_data=True  : write N objects to the bucket via s3dlio
  Phase 2 — list verify         : s3dlio.list() confirms expected object count
  Phase 3 — train=True          : read all objects back via s3dlio

Credentials and endpoint are taken from environment variables (loaded from
the repo's .env file by the wrapper shell script).

Usage (via shell script):
    bash tests/object-store/test_s3dlio_formats.sh

Usage (direct, with venv already active):
    python3 tests/object-store/test_s3dlio_formats.py
    python3 tests/object-store/test_s3dlio_formats.py npz hdf5
    DLIO_TEST_FMT=npy,npz python3 tests/object-store/test_s3dlio_formats.py

Exit code: 0 if all formats pass, 1 if any fail.
"""

import os
import sys
import uuid
import logging
import shutil
import traceback
from pathlib import Path

# ─── RUST_LOG must be set before any s3dlio import ────────────────────────────
os.environ.setdefault("RUST_LOG", "info")

# ─── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
# Suppress noisy third-party loggers
for _noisy in ("urllib3", "botocore", "s3transfer", "filelock", "hydra",
               "omegaconf", "absl"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

log = logging.getLogger("s3dlio-format-test")

# ─── Supported formats ────────────────────────────────────────────────────────
ALL_FORMATS = ["npy", "npz", "hdf5", "csv", "parquet", "jpeg", "png", "tfrecord"]
# No formats are generate-only: tfrecord now uses TFRecordReaderS3Iterable (s3dlio raw
# bytes) for the read phase, so no tensorflow parsing is required to read back.
# Generation still requires tensorflow (TFRecordGenerator) — install with pip.
GENERATE_ONLY_FORMATS: set = set()

# ─── DLIO imports (after env is set) ──────────────────────────────────────────
import importlib.util
from pathlib import Path
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from dlio_benchmark.utils.config import ConfigArguments
from dlio_benchmark.utils.utility import DLIOMPI
from dlio_benchmark.main import DLIOBenchmark

# dlio_benchmark.__file__ is None for editable installs under Python 3.13
# (namespace package).  Locate the configs/ directory via a known submodule
# (dlio_benchmark.utils) so we get the correct inner package path regardless
# of whether submodule_search_locations points to the repo root or the package.
_utils_spec = importlib.util.find_spec("dlio_benchmark.utils")
_PKG_DIR = Path(next(iter(_utils_spec.submodule_search_locations))).parent
_CONFIG_DIR = str(_PKG_DIR / "configs") + "/"
log.info("dlio_benchmark configs: %s", _CONFIG_DIR)
_OUTPUT_DIR = "/tmp/dlio-fmt-test-output"

# ─── Helpers ──────────────────────────────────────────────────────────────────

def _endpoint():
    return os.environ.get("AWS_ENDPOINT_URL", "https://172.16.1.40:9000")


def _region():
    return os.environ.get("AWS_REGION", "us-east-1")


def _list_objects(uri: str) -> list:
    import s3dlio
    print(f"  list: {uri} ...", flush=True)
    result = s3dlio.list(uri, recursive=True)
    print(f"  list: found {len(result)} object(s)", flush=True)
    for u in result:
        print(f"    {u}", flush=True)
    return result


def _cleanup(bucket: str, prefix: str) -> None:
    import s3dlio
    list_uri = f"s3://{bucket}/{prefix.strip('/')}/"
    print(f"  cleanup: listing {list_uri} ...", flush=True)
    try:
        uris = s3dlio.list(list_uri, recursive=True)
    except Exception as exc:
        print(f"  cleanup: list raised: {exc}", flush=True)
        return
    print(f"  cleanup: deleting {len(uris)} object(s)", flush=True)
    for uri in uris:
        try:
            s3dlio.delete(uri)
            print(f"    deleted: {uri}", flush=True)
        except Exception as exc:
            print(f"  cleanup: delete({uri!r}) raised: {exc}", flush=True)
    print("  cleanup: done", flush=True)


def _base_overrides(bucket: str, prefix: str, fmt: str,
                    num_train: int, num_eval: int) -> list:
    # TFRecord requires framework=tensorflow to pass validate() — the check at
    # config.py:288 rejects data_loader=pytorch with tfrecord.  s3dlio's generic
    # data loader fetches raw bytes for any format including .tfrecord objects;
    # the TFRecordReaderS3Iterable handles reading without tensorflow parsing.
    if fmt == "tfrecord":
        framework = "tensorflow"
        data_loader = "tensorflow"
    else:
        framework = "pytorch"
        data_loader = "pytorch"
    return [
        f"++workload.framework={framework}",
        f"++workload.reader.data_loader={data_loader}",
        "++workload.storage.storage_type=s3",
        f"++workload.storage.storage_root={bucket}",
        "++workload.storage.storage_library=s3dlio",
        f"++workload.storage.storage_options.endpoint_url={_endpoint()}",
        f"++workload.storage.storage_options.region={_region()}",
        f"++workload.dataset.data_folder={prefix}",
        f"++workload.dataset.format={fmt}",
        f"++workload.dataset.num_files_train={num_train}",
        f"++workload.dataset.num_files_eval={num_eval}",
        "++workload.dataset.num_samples_per_file=4",
        "++workload.dataset.record_length=256",
        "++workload.dataset.record_length_stdev=0",
        "++workload.dataset.num_subfolders_train=0",
        "++workload.dataset.num_subfolders_eval=0",
    ]


def _run_benchmark(workload_dict: dict, phase: str) -> None:
    log.info("  [%s] DLIOBenchmark.initialize ...", phase)
    workload_dict.setdefault("output", {})["folder"] = _OUTPUT_DIR
    ConfigArguments.reset()
    bench = DLIOBenchmark(workload_dict)
    bench.initialize()
    log.info("  [%s] DLIOBenchmark.run ...", phase)
    bench.run()
    log.info("  [%s] DLIOBenchmark.finalize ...", phase)
    bench.finalize()
    log.info("  [%s] complete", phase)


def _banner(msg: str, width: int = 60) -> None:
    print(f"\n{'═' * width}")
    print(f"  {msg}")
    print(f"{'═' * width}")


# ─── Per-format test ──────────────────────────────────────────────────────────

def run_format_test(fmt: str, bucket: str, num_train: int = 4, num_eval: int = 2) -> bool:
    """
    Run the full put+verify+get cycle for a single format.
    Returns True on success, False on failure.
    """
    # Reset DLIOMPI before each format so a previous run's child-process
    # worker_init(-1) call (which leaves the singleton in CHILD_INITIALIZED
    # state) doesn't prevent re-initialization for the next format.
    DLIOMPI.reset()
    # MPI must be initialized before DLIOBenchmark is constructed.
    DLIOMPI.get_instance().initialize()

    generate_only = fmt in GENERATE_ONLY_FORMATS
    run_id = str(uuid.uuid4())[:8]
    prefix = f"dlio-fmt-test/{run_id}/{fmt}"

    _banner(f"Format: {fmt.upper()}  (run_id={run_id})")
    log.info("bucket=%s  prefix=%s  endpoint=%s", bucket, prefix, _endpoint())
    if generate_only:
        log.info("NOTE: %s is generate-only (no read phase)", fmt)

    base = _base_overrides(bucket, prefix, fmt, num_train, num_eval)

    try:
        # Hydra's global config store must be cleared between test invocations.
        GlobalHydra.instance().clear()

        with initialize_config_dir(version_base=None, config_dir=_CONFIG_DIR):

            # ── Phase 1: Generate (put) ───────────────────────────────────────
            log.info("Phase 1: generate_data → writing %d train + %d eval objects ...",
                     num_train, num_eval)
            cfg = compose(config_name="config", overrides=base + [
                "++workload.workflow.generate_data=True",
                "++workload.workflow.train=False",
                "++workload.workflow.checkpoint=False",
            ])
            _run_benchmark(OmegaConf.to_container(cfg["workload"], resolve=True),
                           phase="datagen")

            # ── Phase 2: Verify (list) ────────────────────────────────────────
            print(f"\n--- Phase 2: verifying objects in bucket ---", flush=True)
            train_uri = f"s3://{bucket}/{prefix}/train/"
            valid_uri = f"s3://{bucket}/{prefix}/valid/"

            found_train = _list_objects(train_uri)
            found_valid = _list_objects(valid_uri)

            if len(found_train) != num_train:
                raise AssertionError(
                    f"Expected {num_train} train objects, found {len(found_train)}"
                )
            if len(found_valid) != num_eval:
                raise AssertionError(
                    f"Expected {num_eval} valid objects, found {len(found_valid)}"
                )
            print(f"  Phase 2: PASSED — {len(found_train)} train + {len(found_valid)} valid objects confirmed", flush=True)

            # ── Phase 3: Read (get) — skipped for generate-only formats ───────
            if not generate_only:
                print(f"\n--- Phase 3: train → reading {num_train} train + {num_eval} valid objects back ---", flush=True)
                cfg = compose(config_name="config", overrides=base + [
                    "++workload.workflow.generate_data=False",
                    "++workload.workflow.train=True",
                    "++workload.workflow.checkpoint=False",
                    "++workload.train.epochs=1",
                    "++workload.train.computation_time=0.0",
                    "++workload.reader.read_threads=0",  # main thread only — avoids fork() deadlock
                    "++workload.reader.batch_size=2",
                ])
                _run_benchmark(OmegaConf.to_container(cfg["workload"], resolve=True),
                               phase="train")
                print("  Phase 3: PASSED — all objects read back successfully", flush=True)

        print(f"\n  ✓ {fmt.upper()} PASSED", flush=True)
        return True

    except Exception as exc:
        print(f"\n  ✗ {fmt.upper()} FAILED: {exc}", flush=True)
        traceback.print_exc()
        return False

    finally:
        # Always clean up test objects
        _cleanup(bucket, prefix)
        shutil.rmtree(_OUTPUT_DIR, ignore_errors=True)
        GlobalHydra.instance().clear()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    bucket = os.environ.get("DLIO_TEST_BUCKET", "mlp-s3dlio")

    # Determine which formats to test:
    #   1. Command-line arguments (highest priority)
    #   2. DLIO_TEST_FMT env var (comma-separated)
    #   3. All formats
    if len(sys.argv) > 1:
        formats = sys.argv[1:]
    elif os.environ.get("DLIO_TEST_FMT"):
        formats = [f.strip() for f in os.environ["DLIO_TEST_FMT"].split(",")]
    else:
        formats = ALL_FORMATS

    unknown = [f for f in formats if f not in ALL_FORMATS]
    if unknown:
        log.error("Unknown format(s): %s  (valid: %s)", unknown, ALL_FORMATS)
        return 1

    _banner(f"s3dlio Format Tests  —  {len(formats)} format(s)", width=60)
    print(f"  Formats  : {', '.join(formats)}")
    print(f"  Bucket   : {bucket}")
    print(f"  Endpoint : {_endpoint()}")
    print(f"  RUST_LOG : {os.environ.get('RUST_LOG', '(unset)')}")
    print()

    results: dict[str, bool] = {}
    for fmt in formats:
        results[fmt] = run_format_test(fmt, bucket)

    # ── Summary ───────────────────────────────────────────────────────────────
    _banner("RESULTS SUMMARY", width=60)
    passed = [f for f, ok in results.items() if ok]
    failed = [f for f, ok in results.items() if not ok]
    for fmt, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {fmt}")
    print()
    print(f"  {len(passed)}/{len(results)} passed")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
