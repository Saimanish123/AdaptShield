#!/usr/bin/env python3
"""Tool-aware AdaptShield baseline for world-modeling demos."""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

from baseline import (
    BENCHMARK,
    MAX_STEPS,
    POLICY,
    TASKS,
    action_from_payload,
    log_end,
    log_step,
    phase1_payload as no_tool_phase1_payload,
    phase2_payload as no_tool_phase2_payload,
    print_replay,
)
from server.adaptshield_environment import AdaptShieldEnvironment
from soc_tools import infer_threat_from_tool_results, investigate_local


MODEL_NAME = "tool-aware-baseline"

def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def phase2_payload(obs: Any, belief_by_turn: Dict[int, Dict[str, str]]) -> Dict[str, Any]:
    """Use belief inferred from observable SOC tool evidence when Phase 2 is ambiguous."""
    belief = belief_by_turn.get(int(obs.turn), {})
    if obs.task_name == "polymorphic-zero-day" and belief:
        return {
            "action": belief["action"],
            "target_node": belief["target_node"],
            "reasoning": "inferred from observable SOC tool fields",
        }

    return no_tool_phase2_payload(obs)


def phase1_payload(obs: Any, belief_by_turn: Dict[int, Dict[str, str]]) -> Dict[str, Any]:
    """Use tool-derived belief in Phase 1 so the baseline is tool-aware end to end."""
    belief = belief_by_turn.get(int(obs.turn), {})
    if obs.task_name == "polymorphic-zero-day" and belief:
        return {
            "threat_type": belief["threat_type"],
            "confidence": 0.86,
            "target_node": belief["target_node"],
            "recommended_action": belief["action"],
            "reasoning": "classified from observable SOC tool fields",
        }

    return no_tool_phase1_payload(obs)

def run_task(task: str, emit_logs: bool = True) -> Dict[str, Any]:
    env = AdaptShieldEnvironment(task_name=task)
    obs = env.reset()
    rewards: List[float] = []
    steps = 0
    belief_by_turn: Dict[int, Dict[str, str]] = {}

    if emit_logs:
        log_start(task)

    while not obs.done and steps < MAX_STEPS:
        if obs.phase == 1:
            tool_results = investigate_local(env, obs, use_tools=True)
            belief_by_turn[int(obs.turn)] = infer_threat_from_tool_results(tool_results)
            payload = phase1_payload(obs, belief_by_turn)
        else:
            payload = phase2_payload(obs, belief_by_turn)

        obs = env.step(action_from_payload(payload))
        reward = float(obs.reward)
        rewards.append(reward)
        steps += 1

        if emit_logs:
            log_step(steps, payload, reward, obs.done)

    metadata = obs.metadata if isinstance(obs.metadata, dict) else {}
    score = float(metadata.get("normalized_score", 0.01))
    success = obs.done and 0.01 <= score <= 0.99

    if emit_logs:
        log_end(success, steps, score, rewards)
        tool_trace = metadata.get("tool_trace") or []
        print(f"[TOOLS] calls={len(tool_trace)} trace={json.dumps(tool_trace[-4:], separators=(',', ':'))}")

    return {
        "task": task,
        "score": score,
        "steps": steps,
        "done": bool(obs.done),
        "rewards": rewards,
        "metadata": metadata,
        "normalized_score_present": "normalized_score" in metadata,
        "success": success,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AdaptShield tool-aware baseline.")
    parser.add_argument("--task", default="polymorphic-zero-day", choices=TASKS + ["all"])
    parser.add_argument("--replay", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = TASKS if args.task == "all" else [args.task]

    for index, task in enumerate(tasks):
        if index:
            print()
        result = run_task(task, emit_logs=True)
        if args.replay:
            print_replay(task, result["metadata"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
