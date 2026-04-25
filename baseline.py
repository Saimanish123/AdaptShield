#!/usr/bin/env python3
"""Rule-based AdaptShield baseline with evaluator-style stdout."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List


REPO_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = REPO_ROOT / "adaptshield"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from models import AdaptShieldAction
from server.adaptshield_environment import AdaptShieldEnvironment


TASKS = ["direct-triage", "dual-pivot", "polymorphic-zero-day"]
BENCHMARK = "adaptshield"
MODEL_NAME = "rule-baseline"
MAX_STEPS = 30

POLICY = {
    "brute_force": ("auth_service", "rate_limit"),
    "lateral_movement": ("payment_service", "isolate"),
    "exfiltration": ("database", "honeypot"),
    "supply_chain": ("api_gateway", "patch"),
    "benign": ("api_gateway", "monitor"),
}


def log_start(task: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={MODEL_NAME}", flush=True)


def log_step(step: int, action: Dict[str, Any], reward: float, done: bool) -> None:
    action_str = json.dumps(action, separators=(",", ":"))
    if len(action_str) > 100:
        action_str = action_str[:97] + "..."
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={str(done).lower()} error=null",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    reward_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={reward_str}",
        flush=True,
    )


def print_replay(task: str, metadata: Dict[str, Any]) -> None:
    replay = metadata.get("episode_replay") or []
    print()
    print(f"Replay: {task}")
    if not replay:
        print("  No replay metadata available.")
        return

    for row in replay:
        shift = " STRATEGY SHIFT:" if row.get("shift") else ""
        foothold = " FOOTHOLD:" if row.get("foothold_transition") else ""
        blast_radius = row.get("blast_radius") or []
        blast = ",".join(blast_radius) if blast_radius else "none"
        active_defenses = row.get("active_defenses") or []
        print(
            f"  Turn {row.get('turn')}:"
            f"{shift}{foothold} {row.get('p1')} -> {row.get('p2_action')}"
            f"({row.get('target', 'unknown')}) -> {row.get('result')}"
            f" | impact={float(row.get('impact', 0.0)):.2f}"
            f" blast_radius={blast}"
            f" active_controls={len(active_defenses)}"
            f" mission={row.get('mission_alignment', 'neutral')}"
        )


def classify_from_metrics(network_nodes: Dict[str, Dict[str, Any]]) -> str:
    auth = network_nodes.get("auth_service", {})
    payment = network_nodes.get("payment_service", {})
    database = network_nodes.get("database", {})
    gateway = network_nodes.get("api_gateway", {})

    if float(auth.get("error_rate", 0.0)) >= 0.10:
        return "brute_force"
    if payment.get("status") == "suspicious" or float(payment.get("cpu", 0)) >= 55:
        return "lateral_movement"
    if float(database.get("outbound_mb", 0)) >= 50:
        return "exfiltration"
    if gateway.get("status") == "suspicious":
        return "supply_chain"
    return "benign"


def phase1_payload(obs) -> Dict[str, Any]:
    threat_type = classify_from_metrics(obs.network_nodes)
    target_node, action = POLICY[threat_type]
    return {
        "threat_type": threat_type,
        "confidence": 0.90,
        "target_node": target_node,
        "recommended_action": action,
        "reasoning": "rule-based metric classifier",
    }


def phase2_payload(obs) -> Dict[str, Any]:
    assessment = obs.phase1_assessment or {}
    threat_type = str(assessment.get("threat_type", "benign"))
    fallback_target, fallback_action = POLICY.get(threat_type, POLICY["benign"])
    action = str(assessment.get("recommended_action") or fallback_action)
    target_node = str(assessment.get("target_node") or fallback_target)
    return {
        "action": action,
        "target_node": target_node,
        "reasoning": "execute analyst recommendation",
    }


def action_from_payload(payload: Dict[str, Any]) -> AdaptShieldAction:
    return AdaptShieldAction(**payload)


def run_task(task: str, emit_logs: bool = True) -> Dict[str, Any]:
    env = AdaptShieldEnvironment(task_name=task)
    obs = env.reset()
    rewards: List[float] = []
    steps = 0

    if emit_logs:
        log_start(task)

    while not obs.done and steps < MAX_STEPS:
        if obs.phase == 1:
            payload = phase1_payload(obs)
        else:
            payload = phase2_payload(obs)

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
    parser = argparse.ArgumentParser(description="Run AdaptShield rule baseline.")
    parser.add_argument(
        "--task",
        default="direct-triage",
        choices=TASKS + ["all"],
        help="Task to run, or 'all' for every task.",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Print a human-readable final episode replay.",
    )
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
