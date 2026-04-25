#!/usr/bin/env python3
"""Run all AdaptShield tasks with the local rule baseline."""

from __future__ import annotations

from baseline import TASKS, run_task


def status_for(result: dict) -> str:
    score = result["score"]
    passed = (
        result["done"] and
        result["normalized_score_present"] and
        0.01 <= score <= 0.99
    )
    return "PASS" if passed else "FAIL"


def main() -> int:
    results = [run_task(task, emit_logs=False) for task in TASKS]

    print("AdaptShield Evaluation")
    print()
    print(f"{'Task':<24} {'Score':>7} {'Steps':>5} {'normalized_score':>18} {'Status':>8}")
    print("-" * 68)

    for result in results:
        normalized = "yes" if result["normalized_score_present"] else "no"
        print(
            f"{result['task']:<24} "
            f"{result['score']:>7.3f} "
            f"{result['steps']:>5} "
            f"{normalized:>18} "
            f"{status_for(result):>8}"
        )

    scores = [result["score"] for result in results]
    staircase = all(left > right for left, right in zip(scores, scores[1:]))
    print()
    print(f"Difficulty staircase: {'PASS' if staircase else 'FAIL'}")

    return 0 if all(status_for(result) == "PASS" for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
