import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = REPO_ROOT / "adaptshield"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

import __init__ as adaptshield
import server.app as server_app
import train as train_module
from client import AdaptshieldEnv
from models import AdaptShieldAction
from server.adaptshield_environment import AdaptShieldEnvironment
from server.grader import normalize_episode_score, _required_tool_fusion


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
        self.assertGreaterEqual(next_turn_obs.reward, 0.65)
        self.assertIn("requires stronger SOC evidence", next_turn_obs.last_action_result)
        self.assertIn("business_impact", next_turn_obs.metadata["score_breakdown"])
        self.assertIn("dependency_blast_radius", next_turn_obs.metadata["score_breakdown"])
        self.assertIn("mission_alignment", next_turn_obs.metadata["score_breakdown"])
        self.assertIn("active_defenses", next_turn_obs.metadata)
        self.assertIn("available_tools", next_turn_obs.metadata)
        tool_names = {tool["name"] for tool in next_turn_obs.metadata["available_tools"]}
        self.assertTrue({
            "identity_lookup",
            "change_calendar_lookup",
            "netflow_lookup",
        }.issubset(tool_names))

        env = AdaptShieldEnvironment("direct-triage")
        env.reset()
        env.call_tool("log_search", node="auth_service")
        env.step(
            AdaptShieldAction(
                threat_type="brute_force",
                confidence=0.9,
                target_node="auth_service",
                recommended_action="rate_limit",
            )
        )
        verified_obs = env.step(
            AdaptShieldAction(action="rate_limit", target_node="auth_service")
        )
        self.assertGreaterEqual(verified_obs.reward, 0.9)
        self.assertIn("Optimal: rate_limit", verified_obs.last_action_result)

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
        for _ in range(8):
            obs = env.reset()
            turn_config = dict(getattr(env, "_turn_config", {}) or {})
            if not turn_config.get("is_benign", False):
                break
        else:
            self.fail("Expected a non-benign hard-task reset within 8 attempts")

        self.assertIn("available_tools", obs.metadata)
        self.assertNotIn("foothold_established", obs.metadata)

        target = str(turn_config.get("correct_target", "auth_service"))
        for tool_name in sorted(_required_tool_fusion("polymorphic-zero-day", str(turn_config.get("strategy", "benign")))):
            tool_result = env.call_tool(tool_name, node=target)
            self.assertNotIn("verified", tool_result)
            self.assertNotIn("evidence_type", tool_result)
            self.assertTrue(tool_result.get("result_summary"))

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
        self.assertGreaterEqual(obs.reward, 0.65)

    def test_enterprise_context_tools_return_public_fields_only(self) -> None:
        env = AdaptShieldEnvironment("polymorphic-zero-day")
        env.reset()

        identity = env.call_tool("identity_lookup", node="payment_service")
        self.assertIn("account", identity)
        self.assertIn("recent_source_host", identity)
        self.assertNotIn("verified", identity)
        self.assertNotIn("evidence_type", identity)

        change = env.call_tool("change_calendar_lookup", node="api_gateway")
        self.assertIn("scheduled", change)
        self.assertIn("change_status", change)
        self.assertNotIn("verified", change)
        self.assertNotIn("evidence_type", change)

        netflow = env.call_tool("netflow_lookup", node="database")
        self.assertIn("traffic_pattern", netflow)
        self.assertIn("east_west_connections", netflow)
        self.assertNotIn("verified", netflow)
        self.assertNotIn("evidence_type", netflow)

    def test_dual_pivot_requires_tool_confirmation_after_pivot(self) -> None:
        env = AdaptShieldEnvironment("dual-pivot")
        env.reset()

        for _ in range(3):
            turn_config = dict(getattr(env, "_turn_config", {}) or {})
            env.step(
                AdaptShieldAction(
                    threat_type=str(turn_config.get("strategy", "brute_force")),
                    confidence=0.9,
                    target_node=str(turn_config.get("correct_target", "auth_service")),
                    recommended_action=str(turn_config.get("correct_action", "monitor")),
                )
            )
            obs = env.step(
                AdaptShieldAction(
                    action=str(turn_config.get("correct_action", "monitor")),
                    target_node=str(turn_config.get("correct_target", "auth_service")),
                )
            )
            self.assertFalse(obs.done)

        turn_config = dict(getattr(env, "_turn_config", {}) or {})
        self.assertEqual(turn_config.get("strategy"), "lateral_movement")
        target = str(turn_config.get("correct_target", "payment_service"))

        env.step(
            AdaptShieldAction(
                threat_type="lateral_movement",
                confidence=0.9,
                target_node=target,
                recommended_action=str(turn_config.get("correct_action", "isolate")),
            )
        )
        obs = env.step(
            AdaptShieldAction(
                action=str(turn_config.get("correct_action", "isolate")),
                target_node=target,
            )
        )
        self.assertTrue(obs.metadata["score_breakdown"]["tool_verification_required"])
        self.assertFalse(obs.metadata["score_breakdown"]["tool_evidence_found"])
        self.assertIn("requires stronger SOC evidence", obs.last_action_result)

        env = AdaptShieldEnvironment("dual-pivot")
        env.reset()
        for _ in range(3):
            turn_config = dict(getattr(env, "_turn_config", {}) or {})
            env.step(
                AdaptShieldAction(
                    threat_type=str(turn_config.get("strategy", "brute_force")),
                    confidence=0.9,
                    target_node=str(turn_config.get("correct_target", "auth_service")),
                    recommended_action=str(turn_config.get("correct_action", "monitor")),
                )
            )
            env.step(
                AdaptShieldAction(
                    action=str(turn_config.get("correct_action", "monitor")),
                    target_node=str(turn_config.get("correct_target", "auth_service")),
                )
            )

        turn_config = dict(getattr(env, "_turn_config", {}) or {})
        target = str(turn_config.get("correct_target", "payment_service"))
        env.call_tool("edr_status", node=target)
        env.call_tool("log_search", node=target)
        env.call_tool("identity_lookup", node=target)
        env.step(
            AdaptShieldAction(
                threat_type="lateral_movement",
                confidence=0.9,
                target_node=target,
                recommended_action=str(turn_config.get("correct_action", "isolate")),
            )
        )
        obs = env.step(
            AdaptShieldAction(
                action=str(turn_config.get("correct_action", "isolate")),
                target_node=target,
            )
        )
        self.assertTrue(obs.metadata["score_breakdown"]["tool_verification_required"])
        self.assertTrue(obs.metadata["score_breakdown"]["tool_evidence_found"])
        self.assertIn("Optimal: isolate", obs.last_action_result)

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

    def test_normalized_score_uses_reachable_reward_ceiling(self) -> None:
        rewards = [0.99] * 10
        self.assertEqual(normalize_episode_score(rewards), 0.99)


if __name__ == "__main__":
    unittest.main()
