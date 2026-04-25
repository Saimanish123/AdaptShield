#!/usr/bin/env python3
"""
Quick repo-root smoke test for AdaptShield.

Run from the repo root:
    python smoke_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import __init__ as adaptshield
import server.app as server_app
from models import AdaptShieldAction
from server.adaptshield_environment import AdaptShieldEnvironment


def main() -> int:
    print("AdaptShield smoke test")
    print(f"- package exports: {adaptshield.__all__}")
    print(f"- server app type: {server_app.app.__class__.__name__}")

    env = AdaptShieldEnvironment("direct-triage")
    obs = env.reset()
    print(
        f"- reset: phase={obs.phase} turn={obs.turn} "
        f"score={obs.metadata.get('normalized_score')}"
    )

    obs = env.step(
        AdaptShieldAction(
            threat_type="brute_force",
            confidence=0.9,
            target_node="auth_service",
            recommended_action="rate_limit",
        )
    )
    print(f"- phase 1 -> phase 2: assessment={obs.phase1_assessment}")

    obs = env.step(AdaptShieldAction(action="rate_limit", target_node="auth_service"))
    print(
        f"- phase 2 -> next turn: reward={obs.reward} done={obs.done} "
        f"result={obs.last_action_result}"
    )

    print("Smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
