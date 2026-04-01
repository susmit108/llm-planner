from __future__ import annotations

import argparse
import json

from medical_graph import MissingDependencyError, train_medqa_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the MedQA graph model and build retrieval artifacts."
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--hash-dim", type=int, default=384)
    parser.add_argument("--dropout", type=float, default=0.15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = train_medqa_graph(
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            hidden_dim=args.hidden_dim,
            hash_dim=args.hash_dim,
            dropout=args.dropout,
        )
    except MissingDependencyError as exc:
        print(str(exc))
        return 1

    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
