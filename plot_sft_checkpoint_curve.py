#!/usr/bin/env python3
"""Plot an SFT checkpoint curve with an optional honest baseline start point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_baseline(benchmark: dict[str, Any], key: str) -> float:
    values = benchmark.get(key, {})
    numeric = [float(value) for value in values.values() if value is not None]
    if not numeric:
        raise ValueError(f"No numeric values found under benchmark key '{key}'")
    return sum(numeric) / len(numeric)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot SFT checkpoint learning curve with optional baseline point.")
    parser.add_argument("--metrics", required=True, help="Path to sft_metrics.json")
    parser.add_argument("--output", required=True, help="Where to write the PNG")
    parser.add_argument(
        "--baseline-json",
        default="",
        help="Optional benchmark_table.json path used to prepend a real baseline point.",
    )
    parser.add_argument(
        "--baseline-key",
        default="tool_baseline",
        choices=["tool_baseline", "no_tool_baseline"],
        help="Which benchmark JSON field to average for the prepended baseline point.",
    )
    parser.add_argument(
        "--baseline-label",
        default="baseline",
        help="X-axis label for the prepended baseline point.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    metrics = load_json(Path(args.metrics))
    rows = metrics.get("reward_curve_rows", []) or []
    if not rows:
        raise SystemExit("No reward_curve_rows found in the provided SFT metrics file.")

    labels = [str(row["checkpoint"]) for row in rows]
    train_scores = [float(row["in_distribution_score"]) for row in rows]
    heldout_scores = [float(row["heldout_score"]) for row in rows]

    if args.baseline_json:
        benchmark = load_json(Path(args.baseline_json))
        baseline_value = mean_baseline(benchmark, args.baseline_key)
        labels = [args.baseline_label] + labels
        train_scores = [baseline_value] + train_scores
        heldout_scores = [baseline_value] + heldout_scores

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(f"matplotlib is required to plot this curve: {exc}") from exc

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(11, 5))
    plt.plot(labels, train_scores, marker="o", linewidth=2, color="#174c7a", label="train family")
    plt.plot(labels, heldout_scores, marker="s", linewidth=2, color="#6d4acb", label="held-out family")
    plt.title("Janus SFT Checkpoint Learning Curve")
    plt.xlabel("Checkpoint")
    plt.ylabel("normalized_score")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
