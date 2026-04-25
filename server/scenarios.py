"""
AdaptShield Scenarios

Observation builder and system prompts.
IMPORTANT: No MITRE codes in alerts. No explicit strategy names.
Agent must reason from raw metrics — not pattern match on codes.
"""

from typing import Any, Dict, List

VALID_NODES   = ["auth_service", "payment_service", "database", "api_gateway"]
VALID_ACTIONS = ["rate_limit", "isolate", "honeypot", "patch", "monitor"]

TASK_CONFIGS = {
    "direct-triage": {
        "max_turns":   5,
        "description": "Single fixed attacker strategy. Learn baseline threat response.",
        "mission_profile": {
            "mission_id": "login_stability",
            "primary_asset": "auth_service",
            "sla_priority": "availability",
            "risk_tolerance": "medium",
            "objective": "Preserve user login availability while stopping credential abuse.",
        },
    },
    "dual-pivot": {
        "max_turns":   6,
        "description": "Two strategies alternating every 20 episodes.",
        "mission_profile": {
            "mission_id": "checkout_continuity",
            "primary_asset": "payment_service",
            "sla_priority": "availability",
            "risk_tolerance": "medium",
            "objective": "Keep checkout online while containing internal movement.",
        },
    },
    "polymorphic-zero-day": {
        "max_turns":   8,
        "description": "All four strategies with mid-episode shift and noise injection.",
        "mission_profile": {
            "mission_id": "breach_containment",
            "primary_asset": "database",
            "sla_priority": "containment",
            "risk_tolerance": "low",
            "objective": "Protect sensitive data and adapt response if attacker persistence is established.",
        },
    },
}

PHASE1_SYSTEM = """You are a Threat Analyst defending a 4-node enterprise network.
Analyze the network metrics and security alerts to classify the threat.

Network nodes: auth_service, payment_service, database, api_gateway

Attack strategies to recognize:
- brute_force: High failed login rate on auth_service, error spikes
- lateral_movement: Unusual cross-node connections, service account anomalies
- exfiltration: Anomalous outbound data volume from database node
- supply_chain: Suspicious update requests or hash mismatches on api_gateway
- benign: Normal operations — scheduled jobs, routine activity

Respond ONLY with valid JSON, no other text:
{
  "threat_type": "<brute_force|lateral_movement|exfiltration|supply_chain|benign>",
  "confidence": <0.0 to 1.0>,
  "target_node": "<auth_service|payment_service|database|api_gateway>",
  "recommended_action": "<rate_limit|isolate|honeypot|patch|monitor>",
  "reasoning": "<one sentence chain of thought>"
}"""

PHASE2_SYSTEM = """You are a Tactical Executor defending a 4-node enterprise network.
You receive a threat assessment from the Threat Analyst and must execute the defense.
You CANNOT see raw network logs — act only on the assessment provided.

Available actions:
- rate_limit: Throttle traffic to node. Light touch, keeps service online. Best for DoS/brute force.
- isolate: Take node completely offline. Stops spread but causes downtime. Use for lateral movement.
- honeypot: Redirect attacker to decoy system. Best for data exfiltration attempts.
- patch: Apply security update. Targeted fix for supply chain attacks.
- monitor: Observe without acting. Use only when genuinely uncertain or event is benign.

Respond ONLY with valid JSON, no other text:
{
  "action": "<rate_limit|isolate|honeypot|patch|monitor>",
  "target_node": "<auth_service|payment_service|database|api_gateway>",
  "reasoning": "<one sentence chain of thought>"
}"""


def build_phase1_obs(
    turn_config: Dict[str, Any],
    history:     List[Dict[str, str]],
    task_name:   str,
    turn:        int,
    max_turns:   int,
    episode_id:  str,
    mission_profile: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build Phase 1 observation — full network state visible."""
    mission_profile = mission_profile or {}
    return {
        "scenario_id":        episode_id,
        "task_name":          task_name,
        "phase":              1,
        "turn":               turn,
        "max_turns":          max_turns,
        "network_nodes":      turn_config["network_nodes"],
        "active_alerts":      turn_config["active_alerts"],
        "attack_stage":       turn_config.get("attack_stage", "none"),
        "history":            history[-3:],
        "phase1_assessment":  None,
        "last_action_result": None,
        "system_context":     _with_mission_context(PHASE1_SYSTEM, mission_profile),
        "available_actions":  VALID_ACTIONS,
        "reward":             0.0,
        "done":               False,
        "metadata":           {
            "episode_id":       episode_id,
            "normalized_score": 0.50,  # always present from step 1
            "mission_profile":  mission_profile,
        },
    }


def build_phase2_obs(
    phase1_output: Dict[str, Any],
    history:       List[Dict[str, str]],
    task_name:     str,
    turn:          int,
    max_turns:     int,
    episode_id:    str,
    current_score: float,
    mission_profile: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Build Phase 2 observation.
    CRITICAL: network_nodes and active_alerts are EMPTY.
    Phase 2 agent is blind to raw state — sees only Phase 1 assessment.
    """
    mission_profile = mission_profile or {}
    return {
        "scenario_id":        episode_id,
        "task_name":          task_name,
        "phase":              2,
        "turn":               turn,
        "max_turns":          max_turns,
        "network_nodes":      {},   # deliberately empty
        "active_alerts":      [],   # deliberately empty
        "attack_stage":       "hidden",
        "history":            history[-3:],
        "phase1_assessment":  phase1_output,
        "last_action_result": None,
        "system_context":     _with_mission_context(PHASE2_SYSTEM, mission_profile),
        "available_actions":  VALID_ACTIONS,
        "reward":             0.0,
        "done":               False,
        "metadata":           {
            "episode_id":       episode_id,
            "normalized_score": current_score,  # always present
            "mission_profile":  mission_profile,
        },
    }


def _with_mission_context(system_prompt: str, mission_profile: Dict[str, Any]) -> str:
    if not mission_profile:
        return system_prompt

    mission = "\n".join([
        "",
        "Mission context:",
        f"- mission_id: {mission_profile.get('mission_id', 'unknown')}",
        f"- primary_asset: {mission_profile.get('primary_asset', 'unknown')}",
        f"- sla_priority: {mission_profile.get('sla_priority', 'balanced')}",
        f"- risk_tolerance: {mission_profile.get('risk_tolerance', 'medium')}",
        f"- objective: {mission_profile.get('objective', 'Balance security and availability.')}",
    ])
    return f"{system_prompt}{mission}"
