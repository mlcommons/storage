#!/usr/bin/python3.9
#!/usr/bin/env python3

from mlpstorage.benchmarks import TrainingBenchmark, VectorDBBenchmark
from mlpstorage.cli import parse_arguments, validate_args, update_args
from mlpstorage.logging import setup_logging, apply_logging_options

logger = setup_logging("MLPerfStorage")


def main():
    args = parse_arguments()
    apply_logging_options(logger, args)

    validate_args(args)
    update_args(args)
    program_switch_dict = dict(
        training=TrainingBenchmark,
        vectordb=VectorDBBenchmark,
    )

    benchmark_class = program_switch_dict.get(args.program)
    benchmark = benchmark_class(args)
    benchmark.run()


if __name__ == "__main__":
    main()
