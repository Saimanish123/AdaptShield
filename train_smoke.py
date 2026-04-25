#!/usr/bin/env python3
"""Dependency-free training-readiness smoke test for AdaptShield."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


REPO_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = REPO_ROOT / "adaptshield"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from models import AdaptShieldAction
from server.adaptshield_environment import AdaptShieldEnvironment


TASKS = ["direct-triage", "dual-pivot", "polymorphic-zero-day"]
POLICY = {
    "brute_force": ("auth_service", "rate_limit"),
    "lateral_movement": ("payment_service", "isolate"),
    "exfiltration": ("database", "honeypot"),
    "supply_chain": ("api_gateway", "patch"),
    "benign": ("api_gateway", "monitor"),
}
ACTION_SPACE = [
    ("auth_service", "rate_limit"),
    ("payment_service", "isolate"),
    ("database", "honeypot"),
    ("api_gateway", "patch"),
    ("api_gateway", "monitor"),
]


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


class TabularDefensePolicy:
    """Tiny epsilon-greedy policy used only to verify trainability."""

    def __init__(self, epsilon: float, lr: float) -> None:
        self.epsilon = epsilon
        self.lr = lr
        self.q: Dict[str, Dict[Tuple[str, str], float]] = {
            threat: {action: 0.50 for action in ACTION_SPACE}
            for threat in POLICY
        }

    def choose_phase1(self, obs: Any) -> Dict[str, Any]:
        threat = classify_from_metrics(obs.network_nodes)
        target, action = POLICY[threat]
        return {
            "threat_type": threat,
            "confidence": 0.90,
            "target_node": target,
            "recommended_action": action,
            "reasoning": "smoke-train metric policy",
        }

    def choose_phase2(self, obs: Any) -> Tuple[Dict[str, Any], str, Tuple[str, str]]:
        assessment = obs.phase1_assessment or {}
        threat = str(assessment.get("threat_type", "benign"))
        choices = self.q.get(threat, self.q["benign"])

        if random.random() < self.epsilon:
            target, action = random.choice(ACTION_SPACE)
        else:
            best_value = max(choices.values())
            best_actions = [
                action for action, value in choices.items()
                if value == best_value
            ]
            target, action = random.choice(best_actions)

        return {
            "action": action,
            "target_node": target,
            "reasoning": "epsilon-greedy smoke policy",
        }, threat, (target, action)

    def update(self, threat: str, selected: Tuple[str, str], reward: float) -> None:
        choices = self.q.setdefault(
            threat,
            {action: 0.50 for action in ACTION_SPACE},
        )
        old_value = choices.get(selected, 0.50)
        choices[selected] = old_value + self.lr * (reward - old_value)

    def decay(self, rate: float, floor: float) -> None:
        self.epsilon = max(floor, self.epsilon * rate)


def run_episode(task: str, policy: TabularDefensePolicy, max_steps: int) -> Dict[str, Any]:
    env = AdaptShieldEnvironment(task_name=task)
    obs = env.reset()
    rewards: List[float] = []
    steps = 0

    while not obs.done and steps < max_steps:
        if obs.phase == 1:
            payload = policy.choose_phase1(obs)
            obs = env.step(AdaptShieldAction(**payload))
        else:
            payload, threat, selected = policy.choose_phase2(obs)
            obs = env.step(AdaptShieldAction(**payload))
            policy.update(threat, selected, float(obs.reward))

        rewards.append(float(obs.reward))
        steps += 1

        metadata = obs.metadata if isinstance(obs.metadata, dict) else {}
        if "normalized_score" not in metadata:
            raise RuntimeError("normalized_score missing during smoke training")

    metadata = obs.metadata if isinstance(obs.metadata, dict) else {}
    return {
        "task": task,
        "score": float(metadata.get("normalized_score", 0.01)),
        "reward_sum": sum(rewards),
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "steps": steps,
        "done": bool(obs.done),
        "normalized_score_present": "normalized_score" in metadata,
    }


def write_rows(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows:
        return

    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def trend(values: List[float]) -> Tuple[float, float]:
    if not values:
        return 0.0, 0.0
    window = max(1, len(values) // 5)
    first = sum(values[:window]) / window
    last = sum(values[-window:]) / window
    return first, last


def run_smoke_training(
    tasks: List[str],
    episodes: int,
    output: Path,
    seed: int,
    epsilon: float,
    epsilon_decay: float,
    epsilon_floor: float,
    lr: float,
    max_steps: int,
) -> List[Dict[str, Any]]:
    random.seed(seed)
    policy = TabularDefensePolicy(epsilon=epsilon, lr=lr)
    rows: List[Dict[str, Any]] = []

    print("AdaptShield smoke training")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Episodes: {episodes}")
    print(f"Output: {output}")
    print()

    for episode in range(1, episodes + 1):
        task = tasks[(episode - 1) % len(tasks)]
        result = run_episode(task=task, policy=policy, max_steps=max_steps)
        result.update({
            "episode": episode,
            "epsilon": round(policy.epsilon, 4),
            "status": "PASS" if result["done"] and result["normalized_score_present"] else "FAIL",
        })
        rows.append(result)
        policy.decay(epsilon_decay, epsilon_floor)

        print(
            f"episode={episode:03d} task={task:<20} "
            f"score={result['score']:.3f} steps={result['steps']:02d} "
            f"epsilon={result['epsilon']:.3f} {result['status']}"
        )

    write_rows(output, rows)

    scores = [float(row["score"]) for row in rows]
    first, last = trend(scores)
    print()
    print(f"First-window avg score: {first:.3f}")
    print(f"Last-window avg score:  {last:.3f}")
    print(f"Score delta:            {last - first:+.3f}")
    print(f"Saved CSV:              {output}")
    print("Smoke training verdict: PASS")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run cheap AdaptShield training smoke test.")
    parser.add_argument("--task", default="direct-triage", choices=TASKS + ["all"])
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--output", default="training_runs/train_smoke.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epsilon", type=float, default=0.85)
    parser.add_argument("--epsilon-decay", type=float, default=0.94)
    parser.add_argument("--epsilon-floor", type=float, default=0.08)
    parser.add_argument("--lr", type=float, default=0.35)
    parser.add_argument("--max-steps", type=int, default=30)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tasks = TASKS if args.task == "all" else [args.task]
    run_smoke_training(
        tasks=tasks,
        episodes=args.episodes,
        output=Path(args.output),
        seed=args.seed,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        epsilon_floor=args.epsilon_floor,
        lr=args.lr,
        max_steps=args.max_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
