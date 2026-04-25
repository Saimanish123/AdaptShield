#!/usr/bin/env python3
"""Build a README-friendly benchmark table from baselines and training metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from baseline import TASKS, run_task as run_no_tool_task
from tool_baseline import run_task as run_tool_task


def rows_to_map(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(row["task"]): row for row in rows}


def load_metrics(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def markdown_table(headers: List[str], rows: List[List[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def fmt(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build AdaptShield benchmark comparison table.")
    parser.add_argument("--sft-metrics", required=True, help="Path to sft_metrics.json")
    parser.add_argument("--grpo-metrics", default="", help="Optional path to GRPO metrics.json")
    parser.add_argument("--output", default="artifacts/benchmark_table.md")
    args = parser.parse_args()

    sft_metrics = load_metrics(Path(args.sft_metrics))
    grpo_metrics = load_metrics(Path(args.grpo_metrics)) if args.grpo_metrics else {}

    no_tool_rows = {task: run_no_tool_task(task, emit_logs=False) for task in TASKS}
    tool_rows = {task: run_tool_task(task, emit_logs=False) for task in TASKS}
    sft_eval = rows_to_map(sft_metrics.get("evaluation_rows", []))
    sft_heldout = rows_to_map(sft_metrics.get("heldout_evaluation_rows", []))
    grpo_eval = rows_to_map(grpo_metrics.get("evaluation_rows", [])) if grpo_metrics else {}
    grpo_heldout = rows_to_map(grpo_metrics.get("heldout_evaluation_rows", [])) if grpo_metrics else {}

    rows: List[List[str]] = []
    for task in TASKS:
        rows.append([
            task,
            fmt(no_tool_rows[task]["score"]),
            fmt(tool_rows[task]["score"]),
            fmt(sft_eval.get(task, {}).get("score")),
            fmt(sft_heldout.get(task, {}).get("score")),
            fmt(grpo_eval.get(task, {}).get("score") if grpo_eval else None),
            fmt(grpo_heldout.get(task, {}).get("score") if grpo_heldout else None),
        ])

    md = markdown_table(
        headers=[
            "Task",
            "No-tool baseline",
            "Tool-aware baseline",
            "SFT (train family)",
            "SFT (held-out family)",
            "GRPO (train family)",
            "GRPO (held-out family)",
        ],
        rows=rows,
    )

    summary = {
        "no_tool_baseline": {task: no_tool_rows[task]["score"] for task in TASKS},
        "tool_baseline": {task: tool_rows[task]["score"] for task in TASKS},
        "sft_train_family": {task: sft_eval.get(task, {}).get("score") for task in TASKS},
        "sft_heldout_family": {task: sft_heldout.get(task, {}).get("score") for task in TASKS},
        "grpo_train_family": {task: grpo_eval.get(task, {}).get("score") for task in TASKS} if grpo_eval else {},
        "grpo_heldout_family": {task: grpo_heldout.get(task, {}).get("score") for task in TASKS} if grpo_heldout else {},
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(md + "\n", encoding="utf-8")
    output_path.with_suffix(".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(md)
    print()
    print(f"Saved markdown table to: {output_path}")
    print(f"Saved JSON summary to: {output_path.with_suffix('.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
