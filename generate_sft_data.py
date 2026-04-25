#!/usr/bin/env python3
"""Generate supervised fine-tuning data directly from AdaptShield rollouts."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from models import AdaptShieldAction
from server.adaptshield_environment import AdaptShieldEnvironment
from train import (
    TASKS,
    _current_reference,
    _teacher_payload,
    attach_tool_results,
    build_messages,
    obs_to_dict,
    render_messages,
    task_for_episode,
)
from soc_tools import investigate_local_with_depth


def build_dataset(
    selected_task: str,
    curriculum: bool,
    use_tools: bool,
    rollout_episodes: int,
    max_steps: int,
    seed: int,
) -> List[Dict[str, Any]]:
    random.seed(seed)
    rows: List[Dict[str, Any]] = []

    for episode in range(1, rollout_episodes + 1):
        task, stage = task_for_episode(
            episode=episode,
            total_episodes=rollout_episodes,
            selected_task=selected_task,
            curriculum=curriculum,
        )
        env = AdaptShieldEnvironment(task_name=task)
        obs = env.reset()
        step_count = 0

        while not obs.done and step_count < max_steps:
            phase = int(getattr(obs, "phase", 1))
            tool_results = investigate_local_with_depth(
                env,
                obs,
                use_tools=use_tools,
                thorough=(task == "polymorphic-zero-day"),
            )
            obs_dict = attach_tool_results(obs_to_dict(obs), tool_results)
            messages = build_messages(obs_dict)
            reference = _current_reference(env)
            teacher_payload = _teacher_payload(phase, reference)
            response_text = json.dumps(teacher_payload, separators=(",", ":"))

            rows.append({
                "task": task,
                "stage": stage,
                "episode": episode,
                "turn": int(getattr(obs, "turn", 0) or 0),
                "phase": phase,
                "attack_stage": reference["stage"],
                "is_benign": bool(reference["is_benign"]),
                "expected_threat_type": reference["threat_type"],
                "expected_target_node": reference["target_node"],
                "expected_action": reference["expected_action"],
                "tool_calls": len(tool_results),
                "messages": messages,
                "response": response_text,
                "text": f"{render_messages(messages)}\n\nASSISTANT:\n{response_text}",
            })

            obs = env.step(AdaptShieldAction(**teacher_payload))
            step_count += 1

    return rows


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_task = {task: 0 for task in TASKS}
    by_phase = {1: 0, 2: 0}
    with_tools = 0

    for row in rows:
        task = str(row.get("task", ""))
        phase = int(row.get("phase", 1) or 1)
        if task in by_task:
            by_task[task] += 1
        by_phase[phase] = by_phase.get(phase, 0) + 1
        if int(row.get("tool_calls", 0) or 0) > 0:
            with_tools += 1

    return {
        "rows": len(rows),
        "task_counts": by_task,
        "phase_counts": by_phase,
        "rows_with_tool_calls": with_tools,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AdaptShield SFT JSONL data")
    parser.add_argument(
        "--task",
        default="all",
        choices=["all", *TASKS],
        help="Task to sample. Use 'all' with --curriculum for mixed data.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=120,
        help="Number of rollout episodes to sample.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=20,
        help="Maximum env steps per episode.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Dataset generation seed.",
    )
    parser.add_argument(
        "--curriculum",
        action="store_true",
        help="Use easy->medium->hard sampling schedule.",
    )
    parser.add_argument(
        "--use-tools",
        action="store_true",
        help="Include SOC tool evidence in prompts where applicable.",
    )
    parser.add_argument(
        "--output",
        default="data/adaptshield_sft.jsonl",
        help="Where to write the JSONL dataset.",
    )
    args = parser.parse_args()

    rows = build_dataset(
        selected_task=args.task,
        curriculum=args.curriculum,
        use_tools=args.use_tools,
        rollout_episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = summarize_rows(rows)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote {len(rows)} rows to {output_path}")
    print(f"Summary saved to {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
