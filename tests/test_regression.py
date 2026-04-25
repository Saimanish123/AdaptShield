import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "adaptshield"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import adaptshield
import server.app as server_app
import train as train_module
from adaptshield.client import AdaptshieldEnv
from adaptshield.models import AdaptShieldAction
from server.adaptshield_environment import AdaptShieldEnvironment


class PackageRegressionTests(unittest.TestCase):
    def test_package_import_exports_expected_symbols(self) -> None:
        self.assertIn("AdaptShieldAction", adaptshield.__all__)
        self.assertIn("AdaptShieldObservation", adaptshield.__all__)
        self.assertIn("AdaptshieldEnv", adaptshield.__all__)

    def test_server_app_imports_fastapi_instance(self) -> None:
        self.assertEqual(server_app.app.__class__.__name__, "FastAPI")


class EnvironmentRegressionTests(unittest.TestCase):
    def test_phase_flow_accepts_both_action_shapes(self) -> None:
        env = AdaptShieldEnvironment("direct-triage")

        phase1_obs = env.reset()
        self.assertEqual(phase1_obs.phase, 1)
        self.assertEqual(phase1_obs.turn, 1)
        self.assertEqual(phase1_obs.metadata["normalized_score"], 0.50)
        self.assertIn("mission_profile", phase1_obs.metadata)

        phase2_obs = env.step(
            AdaptShieldAction(
                threat_type="brute_force",
                confidence=0.9,
                target_node="auth_service",
                recommended_action="rate_limit",
            )
        )
        self.assertEqual(phase2_obs.phase, 2)
        self.assertEqual(phase2_obs.phase1_assessment["recommended_action"], "rate_limit")

        next_turn_obs = env.step(
            AdaptShieldAction(action="rate_limit", target_node="auth_service")
        )
        self.assertEqual(next_turn_obs.phase, 1)
        self.assertGreater(next_turn_obs.reward, 0.9)
        self.assertIn("Optimal: rate_limit", next_turn_obs.last_action_result)
        self.assertIn("business_impact", next_turn_obs.metadata["score_breakdown"])
        self.assertIn("dependency_blast_radius", next_turn_obs.metadata["score_breakdown"])
        self.assertIn("mission_alignment", next_turn_obs.metadata["score_breakdown"])
        self.assertIn("active_defenses", next_turn_obs.metadata)
        self.assertIn("available_tools", next_turn_obs.metadata)

    def test_client_payload_omits_empty_metadata_and_serializes_enums(self) -> None:
        client = AdaptshieldEnv(base_url="http://localhost:7860")

        phase1_payload = client._step_payload(
            AdaptShieldAction(
                threat_type="benign",
                confidence=0.8,
                target_node="auth_service",
                recommended_action="monitor",
            )
        )
        self.assertEqual(
            phase1_payload,
            {
                "threat_type": "benign",
                "confidence": 0.8,
                "target_node": "auth_service",
                "recommended_action": "monitor",
            },
        )

        phase2_payload = client._step_payload(
            AdaptShieldAction(action="rate_limit", target_node="auth_service")
        )
        self.assertEqual(
            phase2_payload,
            {"action": "rate_limit", "target_node": "auth_service"},
        )

    def test_hard_task_records_verified_tool_evidence(self) -> None:
        env = AdaptShieldEnvironment("polymorphic-zero-day")
        obs = env.reset()
        self.assertIn("available_tools", obs.metadata)
        self.assertNotIn("foothold_established", obs.metadata)

        turn_config = dict(getattr(env, "_turn_config", {}) or {})
        target = str(turn_config.get("correct_target", "auth_service"))
        tool_result = env.call_tool("log_search", node=target)
        self.assertNotIn("verified", tool_result)
        self.assertNotIn("evidence_type", tool_result)
        self.assertTrue(tool_result.get("events"))
        env.call_tool("cmdb_lookup", node=target)

        env.step(
            AdaptShieldAction(
                threat_type=turn_config.get("strategy", "brute_force"),
                confidence=0.9,
                target_node=target,
                recommended_action=turn_config.get("correct_action", "monitor"),
            )
        )
        obs = env.step(
            AdaptShieldAction(
                action=turn_config.get("correct_action", "monitor"),
                target_node=target,
            )
        )
        breakdown = obs.metadata["score_breakdown"]
        self.assertTrue(breakdown["tool_verification_required"])
        self.assertTrue(breakdown["tool_evidence_found"])
        self.assertGreaterEqual(obs.reward, 0.85)

    def test_prompt_bank_builds_phase_rows_without_gpu_deps(self) -> None:
        rows = train_module.build_prompt_bank(
            tokenizer=None,
            selected_task="all",
            curriculum=True,
            rollout_episodes=3,
            max_steps=6,
            use_tools=True,
            seed=42,
        )
        self.assertTrue(rows)
        phases = {int(row["phase"]) for row in rows}
        tasks = {str(row["task"]) for row in rows}
        self.assertIn(1, phases)
        self.assertIn(2, phases)
        self.assertTrue(tasks.intersection({"direct-triage", "dual-pivot", "polymorphic-zero-day"}))
        hard_rows = [row for row in rows if row["task"] == "polymorphic-zero-day"]
        self.assertTrue(hard_rows)
        self.assertTrue(any(int(row["tool_calls"]) >= 2 for row in hard_rows))


if __name__ == "__main__":
    unittest.main()
