#!/usr/bin/env python3
"""Plot AdaptShield training CSV or metrics JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple


def load_scores(path: Path) -> Tuple[List[int], List[float], str, List[str]]:
    if path.suffix == ".json":
        data = json.loads(path.read_text())
        rows = data.get("rows", []) or data.get("evaluation_rows", [])
        episodes = [int(row["episode"]) for row in rows]
        scores = [float(row["score"]) for row in rows]
        stages = [str(row.get("stage", row.get("task", ""))) for row in rows]
        return episodes, scores, str(data.get("model", "adaptshield")), stages

    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    episodes = [int(row["episode"]) for row in rows]
    scores = [float(row["score"]) for row in rows]
    stages = [str(row.get("stage", row.get("task", ""))) for row in rows]
    return episodes, scores, "adaptshield-smoke", stages


def moving_average(values: List[float], window: int) -> List[float]:
    smoothed = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        chunk = values[start:index + 1]
        smoothed.append(sum(chunk) / len(chunk))
    return smoothed


def plot(path: Path, output: Path) -> None:
    episodes, scores, label, stages = load_scores(path)
    if not scores:
        raise SystemExit("No scores found to plot.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        first = sum(scores[:max(1, len(scores) // 5)]) / max(1, len(scores) // 5)
        last = sum(scores[-max(1, len(scores) // 5):]) / max(1, len(scores) // 5)
        print("matplotlib is not installed; skipping PNG generation.")
        print(f"Episodes: {len(scores)}")
        print(f"First-window avg: {first:.3f}")
        print(f"Last-window avg:  {last:.3f}")
        print(f"Delta:            {last - first:+.3f}")
        return

    window = max(1, min(10, len(scores) // 5))
    smoothed = moving_average(scores, window)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(episodes, scores, color="#6b8fbf", alpha=0.35, label="raw score")
    ax.plot(episodes, smoothed, color="#123c69", linewidth=2.5, label=f"{window}-episode avg")
    for episode, stage in stage_boundaries(episodes, stages):
        ax.axvline(episode, color="#c44e52", linestyle="--", alpha=0.45)
        ax.text(episode, 0.04, stage.replace("curriculum:", ""), rotation=90, fontsize=8, color="#7a1f24")
    ax.set_title(f"AdaptShield Training Curve ({label})")
    ax.set_xlabel("Episode")
    ax.set_ylabel("normalized_score")
    ax.set_ylim(0.0, 1.0)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    print(f"Saved plot: {output}")


def stage_boundaries(episodes: List[int], stages: List[str]) -> List[Tuple[int, str]]:
    if not stages:
        return []

    boundaries = []
    previous = stages[0]
    for episode, stage in zip(episodes, stages):
        if stage != previous:
            boundaries.append((episode, stage))
            previous = stage
    return boundaries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot AdaptShield training output.")
    parser.add_argument("--input", default="training_runs/train_smoke.csv")
    parser.add_argument("--output", default="training_runs/reward_curve.png")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plot(Path(args.input), Path(args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
