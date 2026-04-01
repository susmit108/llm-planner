from __future__ import annotations

import argparse
import json

from medical_graph import MissingDependencyError, benchmark_saved_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark saved MedQA graph artifacts on the MedQA test split."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional number of test examples to evaluate for a quick smoke test.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        results = benchmark_saved_model(limit=args.limit)
    except MissingDependencyError as exc:
        print(str(exc))
        return 1

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
