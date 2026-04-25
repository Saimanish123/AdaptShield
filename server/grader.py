"""
AdaptShield Grader

Fully deterministic reward computation.
No NLP, no LLM-as-judge, no external API calls.
Runs in milliseconds.

INVARIANT: All return values are strictly in [0.01, 0.99].
normalized_score is ALWAYS present in metadata — never missing.
"""

from typing import Any, Dict, List, Tuple


# ── Reward constants ───────────────────────────────────────────────────────
BASE_REWARD         =  0.50   # survival — agent gets this for staying alive
P1_TYPE_BONUS       =  0.15   # Phase 1 correctly identified threat type
P1_TARGET_BONUS     =  0.10   # Phase 1 correctly identified target node
P2_OPTIMAL          =  0.39   # correct + efficient action (0.50+0.39 = 0.89 base max)
P2_HEAVY            =  0.18   # correct action but heavy-handed
P2_UNVERIFIED       =  0.10   # right hard-task action without investigative evidence
P2_WRONG            = -0.25   # wrong action on real threat
FALSE_POSITIVE      = -0.39   # acted aggressively on benign event
STAGE_ESCALATION    = -0.10   # penalty applied when attack stage escalates
CATASTROPHIC        = -0.49   # database exfiltration completed → done=True

# Optimal vs heavy-handed action mapping
OPTIMAL_ACTION = {
    "brute_force":      "rate_limit",
    "lateral_movement": "isolate",
    "exfiltration":     "honeypot",
    "supply_chain":     "patch",
}

HEAVY_ACTION = {
    "brute_force":      "isolate",    # works but unnecessary downtime
    "lateral_movement": "honeypot",   # works but slower
    "exfiltration":     "isolate",    # works but database goes offline
    "supply_chain":     "isolate",    # works but gateway goes offline
}

# Catastrophic: exfiltration succeeds if agent uses wrong action 3+ times
EXFIL_CATASTROPHIC_ACTIONS = ["monitor", "rate_limit", "patch"]

# Operational model: criticality and dependency blast radius.
# This is intentionally lightweight so it strengthens realism without making
# rewards noisy or hard to train against.
ASSET_CRITICALITY = {
    "auth_service":    0.70,
    "payment_service": 0.90,
    "database":        1.00,
    "api_gateway":     0.80,
    "none":            0.00,
    "unknown":         0.50,
}

SERVICE_DEPENDENCIES = {
    "auth_service":    ["payment_service"],
    "payment_service": ["api_gateway"],
    "database":        ["payment_service", "api_gateway"],
    "api_gateway":     ["auth_service", "payment_service", "database"],
    "none":            [],
    "unknown":         [],
}

ACTION_DISRUPTION = {
    "monitor":    0.00,
    "patch":      0.06,
    "rate_limit": 0.10,
    "honeypot":   0.12,
    "isolate":    0.35,
}

MAX_OPERATIONAL_PENALTY = 0.05
MAX_MISSION_ADJUSTMENT = 0.04

BASE_REQUIRED_TOOL_FUSION = {
    "brute_force":      {"log_search", "cmdb_lookup"},
    "lateral_movement": {"edr_status", "log_search"},
    "exfiltration":     {"log_search", "edr_status"},
    "supply_chain":     {"vuln_lookup", "log_search"},
}

TASK_REQUIRED_TOOL_FUSION = {
    "direct-triage": {
        "brute_force": {"log_search"},
    },
    "dual-pivot": {
        "lateral_movement": {"edr_status", "log_search", "identity_lookup"},
    },
    "polymorphic-zero-day": {
        "brute_force": {"log_search", "cmdb_lookup", "identity_lookup"},
        "lateral_movement": {"edr_status", "log_search", "identity_lookup", "cmdb_lookup"},
        "exfiltration": {"log_search", "edr_status", "netflow_lookup", "cmdb_lookup"},
        "supply_chain": {"vuln_lookup", "log_search", "change_calendar_lookup", "cmdb_lookup"},
    },
}


def grade_step(
    phase1_action: Dict[str, Any],
    phase2_action: Dict[str, Any],
    turn_config:   Dict[str, Any],
    stage:         str,
    consecutive_wrong: int,
    task_name: str = "",
    foothold_established: bool = False,
    mission_profile: Dict[str, Any] | None = None,
    tool_context: Dict[str, Any] | None = None,
) -> Tuple[float, bool, Dict[str, Any]]:
    """
    Grade a complete two-phase step.

    Args:
        phase1_action:     Agent's Phase 1 output (threat assessment)
        phase2_action:     Agent's Phase 2 output (defensive action)
        turn_config:       Ground truth from AttackerEngine.build_observation()
        stage:             Current attack stage (recon/exploit/exfiltration)
        consecutive_wrong: How many consecutive wrong actions agent has taken

    Returns:
        (reward, catastrophic_failure, info_dict)
        reward is ALWAYS in range [0.01, 0.99]
    """
    is_benign      = turn_config.get("is_benign", False)
    strategy       = turn_config.get("strategy", "benign")
    correct_action = turn_config.get("correct_action", "monitor")
    correct_target = turn_config.get("correct_target", "none")
    mission_profile = mission_profile or {}
    tool_context = tool_context or {}
    contextual_countermeasure = (
        task_name == "polymorphic-zero-day" and
        foothold_established and
        strategy == "lateral_movement"
    )
    if contextual_countermeasure:
        correct_action = "honeypot"

    p1_type   = str(phase1_action.get("threat_type", ""))
    p1_target = str(phase1_action.get("target_node", ""))
    p2_action = str(phase2_action.get("action", "monitor"))
    p2_target = str(phase2_action.get("target_node", ""))

    reward       = BASE_REWARD
    catastrophic = False

    info = {
        "strategy":          strategy,
        "stage":             stage,
        "correct_action":    correct_action,
        "correct_target":    correct_target,
        "p1_threat_type":    p1_type,
        "p1_target":         p1_target,
        "p2_action":         p2_action,
        "p2_target":         p2_target,
        "is_benign":         is_benign,
        "p1_type_correct":   False,
        "p1_target_correct": False,
        "score_reason":      "",
        "business_impact":   0.0,
        "availability_impact": 0.0,
        "security_risk":     0.0,
        "dependency_blast_radius": [],
        "operational_penalty": 0.0,
        "foothold_established": foothold_established,
        "contextual_countermeasure": contextual_countermeasure,
        "mission_profile": mission_profile,
        "mission_alignment": "neutral",
        "mission_adjustment": 0.0,
        "tool_verification_required": False,
        "tool_evidence_found": False,
        "tool_fusion_required": [],
        "tool_fusion_found": [],
        "tool_count": int(tool_context.get("tool_count", 0) or 0),
    }

    # ── False positive path ────────────────────────────────────────────────
    if is_benign:
        if p2_action in ("rate_limit", "isolate", "honeypot", "patch"):
            reward += FALSE_POSITIVE
            info["acted_correctly"] = False
            info["score_reason"] = (
                f"False positive: used {p2_action} on benign event"
            )
        else:
            reward += 0.15
            info["acted_correctly"] = True
            info["score_reason"] = "Correct: monitored benign event without overreacting"
        _apply_operational_impact(
            info=info,
            action=p2_action,
            target=p2_target,
            result_kind="false_positive" if p2_action != "monitor" else "optimal",
            strategy=strategy,
            stage=stage,
        )
        _apply_mission_alignment(
            info=info,
            action=p2_action,
            target=p2_target,
            result_kind="false_positive" if p2_action != "monitor" else "optimal",
            mission_profile=mission_profile,
        )
        reward += info["mission_adjustment"]
        reward -= info["operational_penalty"]
        return _clamp(reward), False, info

    # ── Phase 1 accuracy bonuses ───────────────────────────────────────────
    if p1_type == strategy:
        reward += P1_TYPE_BONUS
        info["p1_type_correct"] = True

    if p1_target == correct_target:
        reward += P1_TARGET_BONUS
        info["p1_target_correct"] = True

    # ── Catastrophic failure: exfiltration completed ───────────────────────
    if (strategy == "exfiltration" and
            stage == "exfiltration" and
            p2_action in EXFIL_CATASTROPHIC_ACTIONS):
        reward = BASE_REWARD + CATASTROPHIC
        catastrophic = True
        info["score_reason"] = (
            "CATASTROPHIC: Database exfiltration completed — "
            f"agent used {p2_action} instead of honeypot"
        )
        _apply_operational_impact(
            info=info,
            action=p2_action,
            target=p2_target,
            result_kind="wrong",
            strategy=strategy,
            stage=stage,
        )
        _apply_mission_alignment(
            info=info,
            action=p2_action,
            target=p2_target,
            result_kind="wrong",
            mission_profile=mission_profile,
        )
        reward += info["mission_adjustment"]
        return _clamp(reward), catastrophic, info

    # ── Stage escalation penalty ───────────────────────────────────────────
    if stage == "exploit" and consecutive_wrong >= 1:
        reward += STAGE_ESCALATION
    elif stage == "exfiltration" and consecutive_wrong >= 2:
        reward += STAGE_ESCALATION * 2

    # ── Phase 2 action grading ─────────────────────────────────────────────
    optimal = correct_action if contextual_countermeasure else OPTIMAL_ACTION.get(strategy, correct_action)
    heavy   = "" if contextual_countermeasure else HEAVY_ACTION.get(strategy, "")
    requires_tool_verification = (
        not is_benign and
        strategy in OPTIMAL_ACTION and
        (
            task_name == "polymorphic-zero-day" or
            (task_name == "dual-pivot" and strategy == "lateral_movement") or
            (task_name == "direct-triage" and strategy == "brute_force")
        )
    )
    required_tools = _required_tool_fusion(task_name=task_name, strategy=strategy)
    tool_evidence_found, fusion_found = _has_relevant_tool_evidence(
        tool_context=tool_context,
        strategy=strategy,
        target=correct_target,
        required_tools=required_tools,
    )
    info["tool_verification_required"] = requires_tool_verification
    info["tool_evidence_found"] = tool_evidence_found
    info["tool_fusion_required"] = sorted(required_tools)
    info["tool_fusion_found"] = sorted(fusion_found)

    if (
        p2_action == optimal and
        p2_target == correct_target and
        requires_tool_verification and
        not tool_evidence_found
    ):
        reward += P2_UNVERIFIED
        result_kind = "unverified"
        info["score_reason"] = (
            f"Unverified correct action: {p2_action} on {p2_target} would help, "
            f"but {task_name or 'this task'} requires stronger SOC evidence before full credit"
        )

    elif p2_action == optimal and p2_target == correct_target:
        reward += P2_OPTIMAL
        result_kind = "optimal"
        if contextual_countermeasure:
            info["score_reason"] = (
                f"Context-aware optimal: {p2_action} on {p2_target} — "
                "foothold already established, so deception beats isolation"
            )
        else:
            info["score_reason"] = (
                f"Optimal: {p2_action} on {p2_target} — attack stopped efficiently"
            )

    elif p2_action == optimal and p2_target != correct_target:
        reward += P2_HEAVY * 0.5
        result_kind = "wrong_target"
        info["score_reason"] = (
            f"Right action ({p2_action}) but wrong target "
            f"(got {p2_target}, needed {correct_target})"
        )

    elif p2_action == heavy and p2_target == correct_target:
        reward += P2_HEAVY
        result_kind = "heavy"
        info["score_reason"] = (
            f"Heavy-handed: {p2_action} stopped attack on {p2_target} "
            f"but caused unnecessary service disruption"
        )

    else:
        reward += P2_WRONG
        result_kind = "wrong"
        info["score_reason"] = (
            f"Wrong: {p2_action} on {p2_target} — "
            f"needed {correct_action} on {correct_target}"
        )

    acted_correctly = p2_action in (optimal, heavy) and p2_target == correct_target
    info["acted_correctly"] = acted_correctly
    _apply_operational_impact(
        info=info,
        action=p2_action,
        target=p2_target,
        result_kind=result_kind,
        strategy=strategy,
        stage=stage,
    )
    _apply_mission_alignment(
        info=info,
        action=p2_action,
        target=p2_target,
        result_kind=result_kind,
        mission_profile=mission_profile,
    )
    reward += info["mission_adjustment"]
    reward -= info["operational_penalty"]

    return _clamp(reward), catastrophic, info


def _apply_mission_alignment(
    info: Dict[str, Any],
    action: str,
    target: str,
    result_kind: str,
    mission_profile: Dict[str, Any],
) -> None:
    sla_priority = str(mission_profile.get("sla_priority", "balanced"))
    primary_asset = str(mission_profile.get("primary_asset", "unknown"))
    risk_tolerance = str(mission_profile.get("risk_tolerance", "medium"))

    adjustment = 0.0
    alignment = "neutral"

    if sla_priority == "availability" and action == "isolate" and target == primary_asset:
        adjustment -= MAX_MISSION_ADJUSTMENT
        alignment = "sla_violation"
    elif sla_priority == "availability" and result_kind == "optimal" and action in ("rate_limit", "patch", "monitor"):
        adjustment += MAX_MISSION_ADJUSTMENT / 2
        alignment = "sla_aligned"
    elif sla_priority == "containment" and result_kind == "optimal" and action in ("honeypot", "isolate", "patch"):
        adjustment += MAX_MISSION_ADJUSTMENT / 2
        alignment = "containment_aligned"
    elif risk_tolerance == "low" and result_kind in ("wrong", "wrong_target"):
        adjustment -= MAX_MISSION_ADJUSTMENT / 2
        alignment = "risk_misaligned"

    info["mission_alignment"] = alignment
    info["mission_adjustment"] = round(adjustment, 2)


def _apply_operational_impact(
    info: Dict[str, Any],
    action: str,
    target: str,
    result_kind: str,
    strategy: str,
    stage: str,
) -> None:
    """
    Add deterministic business-impact telemetry and a small bounded penalty.

    The penalty is intentionally capped at 0.05 so existing learning curves keep
    their shape while demos can explain service criticality and blast radius.
    """
    criticality = ASSET_CRITICALITY.get(target, ASSET_CRITICALITY["unknown"])
    disruption = ACTION_DISRUPTION.get(action, 0.10)
    dependents = SERVICE_DEPENDENCIES.get(target, [])
    dependency_factor = min(1.0, 0.15 * len(dependents))

    availability = round(min(1.0, disruption * (criticality + dependency_factor)), 2)
    security = _security_risk(result_kind=result_kind, strategy=strategy, stage=stage)
    impact = round(min(1.0, availability + security), 2)

    if result_kind == "optimal":
        penalty = 0.0
    elif result_kind == "unverified":
        penalty = round(min(MAX_OPERATIONAL_PENALTY, impact * MAX_OPERATIONAL_PENALTY / 2), 2)
    else:
        penalty = round(min(MAX_OPERATIONAL_PENALTY, impact * MAX_OPERATIONAL_PENALTY), 2)

    info["business_impact"] = impact
    info["availability_impact"] = availability
    info["security_risk"] = security
    info["dependency_blast_radius"] = dependents if disruption > 0 else []
    info["operational_penalty"] = penalty


def _security_risk(result_kind: str, strategy: str, stage: str) -> float:
    if result_kind in ("optimal", "heavy"):
        return 0.0
    if result_kind == "unverified":
        return 0.08
    if result_kind == "false_positive":
        return 0.0

    stage_risk = {
        "recon": 0.18,
        "exploit": 0.32,
        "exfiltration": 0.50,
    }.get(stage, 0.20)

    if strategy == "exfiltration":
        stage_risk += 0.15
    elif strategy == "lateral_movement":
        stage_risk += 0.08

    return round(min(1.0, stage_risk), 2)


def _has_relevant_tool_evidence(
    tool_context: Dict[str, Any],
    strategy: str,
    target: str,
    required_tools: set[str],
) -> Tuple[bool, set[str]]:
    fusion_found = {
        str(result.get("tool", ""))
        for result in tool_context.get("tool_results", []) or []
        if str(result.get("node", "")) == target
    }
    has_attack_evidence = False
    for evidence in tool_context.get("evidence", []) or []:
        if (
            str(evidence.get("evidence_type", "")) == strategy and
            str(evidence.get("node", "")) == target and
            bool(evidence.get("verified", False))
        ):
            has_attack_evidence = True
            break

    return has_attack_evidence and required_tools.issubset(fusion_found), fusion_found


def _required_tool_fusion(task_name: str, strategy: str) -> set[str]:
    task_rules = TASK_REQUIRED_TOOL_FUSION.get(task_name, {})
    if strategy in task_rules:
        return set(task_rules[strategy])
    return set(BASE_REQUIRED_TOOL_FUSION.get(strategy, set()))


def _clamp(value: float) -> float:
    """Strict bounds: never exactly 0.0 or 1.0."""
    return max(0.01, min(0.99, round(value, 2)))


def normalize_episode_score(rewards: List[float]) -> float:
    """
    Normalize episode rewards to a single score strictly in (0.01, 0.99).
    ALWAYS returns a value — never raises, never returns exactly 0 or 1.
    """
    if not rewards:
        return 0.50

    total = sum(rewards)
    n = len(rewards)

    # Per-step rewards are clamped before they enter the episode reward list,
    # so normalization must use the reachable ceiling instead of the raw
    # unclamped sum of bonuses. Otherwise perfect episodes top out around 0.87.
    max_step_reward = _clamp(
        BASE_REWARD + P2_OPTIMAL + P1_TYPE_BONUS + P1_TARGET_BONUS + MAX_MISSION_ADJUSTMENT
    )
    min_step_reward = _clamp(BASE_REWARD + CATASTROPHIC)
    max_poss = n * max_step_reward
    min_poss = n * min_step_reward

    if max_poss == min_poss:
        return 0.50

    raw = (total - min_poss) / (max_poss - min_poss)
    return _clamp(raw)
