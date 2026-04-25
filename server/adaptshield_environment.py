"""
AdaptShield Environment

Two-phase agentic cybersecurity environment implementing full OpenEnv spec.

Phase 1 (Threat Analyst): Agent reads raw SIEM state, outputs threat assessment.
Phase 2 (Tactical Executor): Agent reads ONLY Phase 1 output, executes defense.

The attacker progresses through stages (recon→exploit→exfiltration) if agent
fails to act. On the hard task, strategy shifts mid-episode after turn 3.

OpenEnv compliance:
- reset() returns initial observation
- step() returns observation with reward, done, info
- state property returns current State
- SUPPORTS_CONCURRENT_SESSIONS = True
- normalized_score ALWAYS present in metadata
"""

import os
import sys
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

from models import AdaptShieldAction, Phase1Action, Phase2Action, AdaptShieldObservation
from server.attacker import AttackerEngine
from server.grader import grade_step, normalize_episode_score, _clamp
from server.scenarios import (
    TASK_CONFIGS,
    build_phase1_obs,
    build_phase2_obs,
    choose_operational_mode,
    choose_world_family,
    mission_profile_for,
)


DEFENSE_TTL = {
    "rate_limit": 2,
    "isolate":    2,
    "honeypot":   3,
    "patch":      4,
}

DEFENSE_SIDE_EFFECT = {
    "rate_limit": "login_latency",
    "isolate":    "service_downtime",
    "honeypot":   "attacker_redirection",
    "patch":      "temporary_restart",
}

AVAILABLE_SOC_TOOLS = [
    {
        "name": "log_search",
        "endpoint": "/tools/log_search",
        "description": "Search recent SIEM/application logs for a node and time window.",
    },
    {
        "name": "cmdb_lookup",
        "endpoint": "/tools/cmdb_lookup",
        "description": "Inspect service ownership, criticality, dependencies, and blast radius.",
    },
    {
        "name": "edr_status",
        "endpoint": "/tools/edr_status",
        "description": "Check endpoint containment, persistence, beaconing, and active controls.",
    },
    {
        "name": "vuln_lookup",
        "endpoint": "/tools/vuln_lookup",
        "description": "Query internal package/advisory risk for supply-chain investigations.",
    },
    {
        "name": "identity_lookup",
        "endpoint": "/tools/identity_lookup",
        "description": "Inspect account type, privilege level, normal host affinity, and anomalous identity use.",
    },
    {
        "name": "change_calendar_lookup",
        "endpoint": "/tools/change_calendar_lookup",
        "description": "Check whether maintenance, deploys, or patch windows were scheduled for the target service.",
    },
    {
        "name": "netflow_lookup",
        "endpoint": "/tools/netflow_lookup",
        "description": "Inspect east-west and outbound traffic summaries for enterprise network pivots and data movement.",
    },
]

SERVICE_OWNERS = {
    "auth_service": "identity-platform",
    "payment_service": "checkout-platform",
    "database": "data-platform",
    "api_gateway": "edge-platform",
}

IDENTITY_CONTEXT = {
    "auth_service": {
        "account": "svc_auth_frontend",
        "account_type": "service_account",
        "privilege_level": "medium",
        "normal_hosts": ["auth_service", "api_gateway"],
    },
    "payment_service": {
        "account": "svc_checkout",
        "account_type": "service_account",
        "privilege_level": "high",
        "normal_hosts": ["payment_service"],
    },
    "database": {
        "account": "svc_data_sync",
        "account_type": "service_account",
        "privilege_level": "high",
        "normal_hosts": ["database", "payment_service"],
    },
    "api_gateway": {
        "account": "deploy_bot",
        "account_type": "automation",
        "privilege_level": "medium",
        "normal_hosts": ["api_gateway"],
    },
}

CHANGE_CALENDAR = {
    "auth_service": {
        "window": "03:00-03:20Z",
        "change_type": "auth policy sync",
        "expected_actor": "svc_auth_frontend",
    },
    "payment_service": {
        "window": "02:30-02:45Z",
        "change_type": "checkout rollout",
        "expected_actor": "svc_checkout",
    },
    "database": {
        "window": "04:00-04:30Z",
        "change_type": "backup and index maintenance",
        "expected_actor": "svc_data_sync",
    },
    "api_gateway": {
        "window": "03:10-03:25Z",
        "change_type": "gateway deploy",
        "expected_actor": "deploy_bot",
    },
}


class AdaptShieldEnvironment(Environment):
    """
    AdaptShield: Two-Phase Adaptive Cybersecurity RL Environment.

    Example:
        >>> env = AdaptShieldEnvironment(task_name="direct-triage")
        >>> obs = env.reset()
        >>> # Phase 1 — classify the threat
        >>> obs2 = env.step(Phase1Action(
        ...     threat_type="brute_force", confidence=0.9,
        ...     target_node="auth_service", recommended_action="rate_limit"
        ... ))
        >>> print(obs2.phase)  # 2
        >>> # Phase 2 — execute the defense
        >>> obs3 = env.step(Phase2Action(
        ...     action="rate_limit", target_node="auth_service"
        ... ))
        >>> print(obs3.reward)  # reward signal
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        task_name: str = "direct-triage",
        world_split: str | None = None,
        world_family: str | None = None,
        operational_mode: str | None = None,
    ):
        if task_name not in TASK_CONFIGS:
            task_name = "direct-triage"

        self._task_name       = task_name
        self._config          = TASK_CONFIGS[task_name]
        self._world_split     = self._sanitize_world_split(world_split or os.environ.get("ADAPTSHIELD_WORLD_SPLIT", "train"))
        self._requested_world_family = world_family or os.environ.get("ADAPTSHIELD_WORLD_FAMILY")
        self._requested_operational_mode = operational_mode or os.environ.get("ADAPTSHIELD_OPERATIONAL_MODE")
        self._world_family    = choose_world_family(self._world_split, self._requested_world_family)
        self._operational_mode = choose_operational_mode(task_name, self._requested_operational_mode)
        self._mission_profile = mission_profile_for(task_name, self._operational_mode, self._world_family)
        self._attacker        = AttackerEngine(task_name, world_family=self._world_family)
        self._state           = State(episode_id=str(uuid4()), step_count=0)

        # Episode state
        self._turn:              int                      = 0
        self._phase:             int                      = 1
        self._rewards:           List[float]              = []
        self._done:              bool                     = False
        self._last_reward:       float                    = 0.0
        self._history:           List[Dict[str, str]]     = []
        self._phase1_output:     Optional[Dict[str, Any]] = None
        self._phase1_grading_output: Optional[Dict[str, Any]] = None
        self._turn_config:       Optional[Dict[str, Any]] = None
        self._consecutive_wrong: int                      = 0
        self._last_obs:          Optional[AdaptShieldObservation] = None
        self._episode_replay:    List[Dict[str, Any]]     = []
        self._last_replay_strategy: Optional[str]          = None
        self._active_defenses:   List[Dict[str, Any]]      = []
        self._foothold_established: bool                   = False
        self._tool_trace:        List[Dict[str, Any]]      = []
        self._turn_tool_evidence: Dict[int, List[Dict[str, Any]]] = {}
        self._turn_tool_results: Dict[int, List[Dict[str, Any]]] = {}

    # ── OpenEnv interface ──────────────────────────────────────────────────

    def reset(self, task_name: str = None) -> AdaptShieldObservation:
        """
        Reset environment. Optionally switch task via task_name.
        Always returns Phase 1 observation (Threat Analyst turn).
        """
        if task_name and task_name in TASK_CONFIGS:
            self._task_name = task_name
            self._config    = TASK_CONFIGS[task_name]
        self._world_family = choose_world_family(self._world_split, self._requested_world_family)
        self._operational_mode = choose_operational_mode(self._task_name, self._requested_operational_mode)
        self._mission_profile = mission_profile_for(self._task_name, self._operational_mode, self._world_family)
        self._attacker  = AttackerEngine(self._task_name, world_family=self._world_family)

        self._state              = State(episode_id=str(uuid4()), step_count=0)
        self._turn               = 1
        self._phase              = 1
        self._rewards            = []
        self._done               = False
        self._last_reward        = 0.0
        self._history            = []
        self._phase1_output      = None
        self._phase1_grading_output = None
        self._consecutive_wrong  = 0
        self._episode_replay     = []
        self._last_replay_strategy = None
        self._active_defenses    = []
        self._foothold_established = False
        self._tool_trace         = []
        self._turn_tool_evidence = {}
        self._turn_tool_results  = {}

        self._attacker.reset_episode()
        self._turn_config = self._prepare_turn_config(self._attacker.build_observation())

        obs_dict = build_phase1_obs(
            turn_config=self._turn_config,
            history=self._history,
            task_name=self._task_name,
            turn=self._turn,
            max_turns=self._config["max_turns"],
            episode_id=self._state.episode_id,
            mission_profile=self._mission_profile,
        )
        obs = self._to_obs(obs_dict)
        obs.metadata = self._metadata_with_defenses(obs.metadata)
        self._last_obs = obs
        return obs

    def step(
        self, action: AdaptShieldAction | Phase1Action | Phase2Action
    ) -> AdaptShieldObservation:  # type: ignore[override]
        """
        Execute one step.

        Accepts either Phase1Action or Phase2Action.
        Phase 1 → transitions to Phase 2 (no reward yet).
        Phase 2 → grades action, advances turn, returns to Phase 1.
        """
        if self._done:
            return self._last_obs or self._error_observation(
                "Episode already completed."
            )

        try:
            self._state.step_count += 1

            # ── Phase 1 → Phase 2 transition ──────────────────────────────
            if self._phase == 1:
                phase1_output = {
                    "threat_type":        _action_value(getattr(action, "threat_type", None), "unknown"),
                    "confidence":         _action_float(getattr(action, "confidence", None), 0.5),
                    "target_node":        _action_value(getattr(action, "target_node", None), "unknown"),
                    "recommended_action": _action_value(getattr(action, "recommended_action", None), "monitor"),
                    "reasoning":          str(getattr(action, "reasoning", "") or ""),
                }
                self._phase1_grading_output = dict(phase1_output)
                self._phase1_output = _degrade_handoff(
                    phase1_output=phase1_output,
                    turn_config=self._turn_config or {},
                    task_name=self._task_name,
                    turn=self._turn,
                )
                self._phase = 2
                current_score = normalize_episode_score(self._rewards)

                obs_dict = build_phase2_obs(
                    phase1_output=self._phase1_output,
                    history=self._history,
                    task_name=self._task_name,
                    turn=self._turn,
                    max_turns=self._config["max_turns"],
                    episode_id=self._state.episode_id,
                    current_score=current_score,
                    mission_profile=self._mission_profile,
                )
                obs = self._to_obs(obs_dict)
                obs.reward = _clamp(self._last_reward if self._last_reward > 0 else 0.01)
                obs.metadata = self._metadata_with_defenses({
                    "episode_id": self._state.episode_id,
                    "normalized_score": float(current_score),
                    "mission_profile": self._mission_profile,
                })
                self._last_obs = obs
                return obs

            # ── Phase 2 — grade and advance turn ──────────────────────────
            p2 = {
                "action":      _action_value(getattr(action, "action", None), "monitor"),
                "target_node": _action_value(getattr(action, "target_node", None), "unknown"),
                "reasoning":   str(getattr(action, "reasoning", "") or ""),
            }

            current_stage = self._attacker.current_stage()
            foothold_before = self._foothold_established
            reward, catastrophic, info = grade_step(
                phase1_action=self._phase1_grading_output or self._phase1_output or {},
                phase2_action=p2,
                turn_config=self._turn_config or {},
                stage=current_stage,
                consecutive_wrong=self._consecutive_wrong,
                task_name=self._task_name,
                foothold_established=foothold_before,
                mission_profile=self._mission_profile,
                tool_context=self._tool_context_for_turn(),
            )

            reward = _clamp(_action_float(reward, 0.01))
            self._register_active_defense(p2)
            foothold_transition = self._update_foothold_state(
                p2=p2,
                info=info,
                stage=current_stage,
            )
            info["foothold_established"] = self._foothold_established
            info["foothold_transition"] = foothold_transition

            # Track consecutive wrong actions for stage escalation
            if info.get("acted_correctly", False):
                self._consecutive_wrong = 0
            else:
                self._consecutive_wrong += 1

            self._rewards.append(reward)
            self._last_reward = reward

            # Update history
            replay_strategy = self._attacker.current_strategy()
            strategy_shift = (
                self._last_replay_strategy is not None and
                replay_strategy != self._last_replay_strategy
            )
            self._last_replay_strategy = replay_strategy
            self._episode_replay.append({
                "turn":        self._turn,
                "p1":          (self._phase1_output or {}).get("threat_type", "unknown"),
                "p2_action":   p2["action"],
                "target":      p2["target_node"],
                "result":      _replay_result(info),
                "shift":       strategy_shift,
                "impact":      float(info.get("business_impact", 0.0)),
                "blast_radius": info.get("dependency_blast_radius", []),
                "active_defenses": self._active_defense_snapshot(),
                "foothold_established": self._foothold_established,
                "foothold_transition": foothold_transition,
                "mission_alignment": info.get("mission_alignment", "neutral"),
                "tool_calls": info.get("tool_count", 0),
                "tool_evidence_found": info.get("tool_evidence_found", False),
            })

            self._history.append({
                "turn":   str(self._turn),
                "p1":     f"classified:{(self._phase1_output or {}).get('threat_type','?')}",
                "p2":     f"{p2['action']}→{p2['target_node']}",
                "result": info.get("score_reason", "")[:80],
                "reward": f"{reward:.2f}",
            })

            # Advance attacker
            self._attacker.advance_turn(
                agent_acted_correctly=info.get("acted_correctly", False)
            )
            self._decay_active_defenses()

            # Advance turn
            self._turn  += 1
            self._phase  = 1
            self._phase1_output = None
            self._phase1_grading_output = None

            episode_done = catastrophic or (self._turn > self._config["max_turns"])
            self._done   = episode_done

            # Compute normalized score — ALWAYS present
            norm_score = normalize_episode_score(self._rewards)

            if not episode_done:
                self._turn_config = self._prepare_turn_config(self._attacker.build_observation())
                obs_dict = build_phase1_obs(
                    turn_config=self._turn_config,
                    history=self._history,
                    task_name=self._task_name,
                    turn=self._turn,
                    max_turns=self._config["max_turns"],
                    episode_id=self._state.episode_id,
                    mission_profile=self._mission_profile,
                )
                obs = self._to_obs(obs_dict)
                obs.reward             = reward
                obs.done               = False
                obs.last_action_result = info.get("score_reason", "")
                obs.metadata = self._metadata_with_defenses({
                    "episode_id":       self._state.episode_id,
                    "normalized_score": float(norm_score),
                    "score_breakdown":  info,
                    "turns_completed":  self._turn - 1,
                    "consecutive_wrong": self._consecutive_wrong,
                    "mission_profile": self._mission_profile,
                })
            else:
                self._attacker.advance_episode()
                obs_dict = build_phase1_obs(
                    turn_config={"network_nodes": {}, "active_alerts": ["[EPISODE COMPLETE]"],
                                 "attack_stage": "none", "is_benign": False,
                                 "strategy": "none", "correct_action": "none", "correct_target": "none"},
                    history=self._history,
                    task_name=self._task_name,
                    turn=self._turn,
                    max_turns=self._config["max_turns"],
                    episode_id=self._state.episode_id,
                    mission_profile=self._mission_profile,
                )
                obs = self._to_obs(obs_dict)
                obs.reward = reward
                obs.done   = True
                obs.last_action_result = info.get("score_reason", "")
                obs.metadata = self._metadata_with_defenses({
                    "episode_id":       self._state.episode_id,
                    "normalized_score": float(norm_score),
                    "score_breakdown":  info,
                    "raw_rewards":      self._rewards,
                    "catastrophic":     catastrophic,
                    "turns_completed":  self._turn - 1,
                    "episode_replay":   self._episode_replay,
                    "mission_profile": self._mission_profile,
                })

            self._last_obs = obs
            return obs
        except Exception as exc:
            return self._error_observation(f"step_error: {exc}")

    @property
    def state(self) -> State:
        """Returns State with episode_id and step_count per OpenEnv spec."""
        return self._state

    # ── Internal ──────────────────────────────────────────────────────────

    def _to_obs(self, d: Dict[str, Any]) -> AdaptShieldObservation:
        return AdaptShieldObservation(
            scenario_id        = d.get("scenario_id", ""),
            task_name          = d.get("task_name", self._task_name),
            phase              = d.get("phase", 1),
            turn               = d.get("turn", 0),
            max_turns          = d.get("max_turns", self._config["max_turns"]),
            network_nodes      = d.get("network_nodes", {}),
            active_alerts      = d.get("active_alerts", []),
            attack_stage       = d.get("attack_stage", "none"),
            history            = d.get("history", []),
            phase1_assessment  = d.get("phase1_assessment"),
            last_action_result = d.get("last_action_result"),
            system_context     = d.get("system_context", ""),
            available_actions  = d.get("available_actions", []),
            reward             = d.get("reward", 0.0),
            done               = d.get("done", False),
            metadata           = d.get("metadata", {"normalized_score": 0.50}),
        )

    @staticmethod
    def _sanitize_world_split(value: str) -> str:
        return value if value in {"train", "eval"} else "train"

    def _error_observation(self, error_message: str) -> AdaptShieldObservation:
        """Return a safe observation instead of letting step() raise."""
        norm_score = float(normalize_episode_score(self._rewards))
        reward = _clamp(self._last_reward if self._last_reward > 0 else 0.01)

        if self._phase == 2:
            obs_dict = build_phase2_obs(
                phase1_output=self._phase1_output or {},
                history=self._history,
                task_name=self._task_name,
                turn=self._turn,
                max_turns=self._config["max_turns"],
                episode_id=self._state.episode_id,
                current_score=norm_score,
                mission_profile=self._mission_profile,
            )
        else:
            turn_config = self._turn_config or {
                "network_nodes": {},
                "active_alerts": [f"[ERROR] {error_message}"],
                "attack_stage": "none",
                "is_benign": False,
                "strategy": "unknown",
                "correct_action": "monitor",
                "correct_target": "unknown",
            }
            obs_dict = build_phase1_obs(
                turn_config=turn_config,
                history=self._history,
                task_name=self._task_name,
                turn=self._turn,
                max_turns=self._config["max_turns"],
                episode_id=self._state.episode_id,
                mission_profile=self._mission_profile,
            )

        obs = self._to_obs(obs_dict)
        obs.reward = float(reward)
        obs.done = bool(self._done)
        obs.last_action_result = error_message
        obs.metadata = self._metadata_with_defenses({
            "episode_id": self._state.episode_id,
            "normalized_score": norm_score,
            "error": error_message,
            "turns_completed": max(0, self._turn - 1),
            "mission_profile": self._mission_profile,
        })
        self._last_obs = obs
        return obs

    def call_tool(self, tool_name: str, **params: Any) -> Dict[str, Any]:
        """
        Query the local SOC tool surface.

        These tools reveal partial evidence, not ground-truth answers. They are
        stateful because responses depend on the current turn, attacker stage,
        foothold state, active defenses, and previous actions.
        """
        try:
            tool_name = str(tool_name or "").strip()
            node = str(params.get("node", params.get("target_node", "unknown")) or "unknown")

            if tool_name == "log_search":
                result = self._tool_log_search(node=node, query=str(params.get("query", "")))
            elif tool_name == "cmdb_lookup":
                result = self._tool_cmdb_lookup(node=node)
            elif tool_name == "edr_status":
                result = self._tool_edr_status(node=node)
            elif tool_name == "vuln_lookup":
                result = self._tool_vuln_lookup(node=node, package=str(params.get("package", "")))
            elif tool_name == "identity_lookup":
                result = self._tool_identity_lookup(node=node)
            elif tool_name == "change_calendar_lookup":
                result = self._tool_change_calendar_lookup(node=node)
            elif tool_name == "netflow_lookup":
                result = self._tool_netflow_lookup(node=node)
            else:
                result = {
                    "tool": tool_name,
                    "turn": self._turn,
                    "node": node,
                    "error": f"Unknown tool: {tool_name}",
                    "available_tools": [tool["name"] for tool in AVAILABLE_SOC_TOOLS],
                }

            self._record_tool_result(result)
            return _public_tool_result(result)
        except Exception as exc:
            return {
                "tool": str(tool_name),
                "turn": self._turn,
                "error": f"tool_error: {exc}",
            }

    def _tool_log_search(self, node: str, query: str = "") -> Dict[str, Any]:
        turn_config = self._turn_config or {}
        hard_task = self._task_name == "polymorphic-zero-day"
        strategy = str(turn_config.get("strategy", "benign"))
        target = str(turn_config.get("correct_target", "none"))
        stage = str(turn_config.get("attack_stage", self._attacker.current_stage()))
        relevant = node == target

        events = [f"{node}: routine health heartbeat accepted"]
        evidence_type = "benign"
        confidence = 0.32

        if relevant and strategy == "brute_force":
            events = [
                "auth_service: 143 failed logins grouped by /24 source subnet",
                "auth_service: password spray pattern across active user accounts",
                "payment_service: no matching checkout errors during same window",
            ]
            evidence_type = "brute_force"
            confidence = 0.88
        elif relevant and strategy == "lateral_movement":
            if hard_task:
                events = [
                    "auth_service: low-volume identity failures persist from earlier abuse window",
                    "payment_service: shared workload credential reused from a new east-west origin",
                    "payment_service: callback mesh fans out across private ports after short-lived sessions",
                ]
                confidence = 0.77
            else:
                events = [
                    "payment_service: service account login from previously unseen host",
                    "auth_service -> payment_service connections exceed baseline",
                    "payment_service: short-lived internal sessions fan out to private ports",
                ]
                confidence = 0.84
            evidence_type = "lateral_movement"
        elif relevant and strategy == "exfiltration":
            if hard_task:
                events = [
                    "database: bulk row access concentrates on high-value records",
                    "database: staging bundle assembled before outbound transfer burst",
                    "netflow: outbound transfer volume from database exceeds 7-day baseline",
                ]
                confidence = 0.82
            else:
                events = [
                    "database: sequential reads concentrated on sensitive tables",
                    "database: compressed archive created before outbound transfer spike",
                    "netflow: database egress exceeds 7-day baseline for this hour",
                ]
                confidence = 0.90
            evidence_type = "exfiltration"
        elif relevant and strategy == "supply_chain":
            if hard_task:
                events = [
                    "api_gateway: release artifact requested from untrusted update channel",
                    "api_gateway: dependency integrity drift detected in gateway manifest",
                    "api_gateway: new outbound connection follows release request",
                ]
                confidence = 0.79
            else:
                events = [
                    "api_gateway: unsigned package update requested by deploy_bot",
                    "api_gateway: dependency hash differs from approved registry hash",
                    "api_gateway: new outbound connection follows update attempt",
                ]
                confidence = 0.86
            evidence_type = "supply_chain"
        elif turn_config.get("is_benign", False):
            events = [
                f"{node}: scheduled maintenance event matches change calendar",
                f"{node}: no persistence, beaconing, or sensitive data access observed",
            ]
            confidence = 0.65

        return {
            "tool": "log_search",
            "turn": self._turn,
            "phase": self._phase,
            "node": node,
            "query": query,
            "events": [self._surface_text(event) for event in events],
            "evidence_type": evidence_type,
            "confidence": confidence,
            "verified": relevant and evidence_type == strategy,
        }

    def _tool_cmdb_lookup(self, node: str) -> Dict[str, Any]:
        dependencies = {
            "auth_service": ["payment_service"],
            "payment_service": ["api_gateway"],
            "database": ["payment_service", "api_gateway"],
            "api_gateway": ["auth_service", "payment_service", "database"],
        }.get(node, [])
        criticality = {
            "auth_service": "high",
            "payment_service": "critical",
            "database": "critical",
            "api_gateway": "high",
        }.get(node, "unknown")
        safe_actions = {
            "auth_service": ["rate_limit", "patch", "monitor"],
            "payment_service": ["honeypot", "rate_limit", "monitor"],
            "database": ["honeypot", "monitor"],
            "api_gateway": ["patch", "rate_limit", "monitor"],
        }.get(node, ["monitor"])

        return {
            "tool": "cmdb_lookup",
            "turn": self._turn,
            "phase": self._phase,
            "node": node,
            "owner": SERVICE_OWNERS.get(node, "unknown"),
            "criticality": criticality,
            "dependencies": dependencies,
            "mission_profile": self._mission_profile,
            "safe_actions": safe_actions,
            "evidence_type": "operational_context",
            "verified": True,
            "result_summary": self._surface_text(f"{node} criticality {criticality}; blast radius {len(dependencies)}"),
        }

    def _tool_edr_status(self, node: str) -> Dict[str, Any]:
        turn_config = self._turn_config or {}
        hard_task = self._task_name == "polymorphic-zero-day"
        strategy = str(turn_config.get("strategy", "benign"))
        target = str(turn_config.get("correct_target", "none"))
        relevant = node == target
        active_controls = [
            control for control in self._active_defenses
            if control.get("target") == node
        ]

        status = {
            "containment": "not_isolated",
            "persistence": False,
            "beaconing": False,
            "active_controls": active_controls,
        }
        evidence_type = "benign"
        confidence = 0.45

        if relevant and strategy == "lateral_movement":
            if hard_task:
                status.update({
                    "containment": "partial" if self._foothold_established else "unconfirmed",
                    "persistence": self._foothold_established,
                    "beaconing": self._foothold_established,
                    "process_note": (
                        "callback process remains active under reused workload credential"
                        if self._foothold_established else
                        "unexpected child process under shared workload credential; no confirmed beacon yet"
                    ),
                })
                confidence = 0.74 if not self._foothold_established else 0.87
            else:
                status.update({
                    "containment": "partial" if self._foothold_established else "not_isolated",
                    "persistence": self._foothold_established,
                    "beaconing": True,
                    "process_note": "unknown child process under service account context",
                })
                confidence = 0.87
            evidence_type = "lateral_movement"
        elif relevant and strategy == "exfiltration":
            status.update({
                "containment": "not_isolated",
                "persistence": self._foothold_established,
                "beaconing": False if hard_task else True,
                "process_note": (
                    "bundle staging followed by intermittent outbound worker activity"
                    if hard_task else
                    "archive process followed by outbound transfer"
                ),
            })
            evidence_type = "exfiltration"
            confidence = 0.73 if hard_task else 0.82
        elif active_controls:
            status["containment"] = "control_active"
            confidence = 0.70

        return {
            "tool": "edr_status",
            "turn": self._turn,
            "phase": self._phase,
            "node": node,
            **status,
            "evidence_type": evidence_type,
            "confidence": confidence,
            "verified": relevant and evidence_type == strategy,
            "process_note": self._surface_text(str(status.get("process_note", ""))) if status.get("process_note") else "",
        }

    def _tool_vuln_lookup(self, node: str, package: str = "") -> Dict[str, Any]:
        turn_config = self._turn_config or {}
        hard_task = self._task_name == "polymorphic-zero-day"
        strategy = str(turn_config.get("strategy", "benign"))
        target = str(turn_config.get("correct_target", "none"))
        relevant = node == target and strategy == "supply_chain"

        if relevant:
            advisory = {
                "package": package or "gateway-router",
                "advisory_id": "ADV-AS-042",
                "risk": "critical",
                "finding": (
                    "artifact integrity drift from untrusted release channel"
                    if hard_task else
                    "registry hash mismatch with unsigned update source"
                ),
                "recommended_mitigation": "patch from trusted registry",
            }
            evidence_type = "supply_chain"
            confidence = 0.82 if hard_task else 0.91
        else:
            advisory = {
                "package": package or "unknown",
                "advisory_id": None,
                "risk": "none_known",
                "finding": "no matching active internal advisory for this node/package",
                "recommended_mitigation": "continue investigation",
            }
            evidence_type = "benign"
            confidence = 0.55

        return {
            "tool": "vuln_lookup",
            "turn": self._turn,
            "phase": self._phase,
            "node": node,
            **{
                **advisory,
                "finding": self._surface_text(str(advisory.get("finding", ""))),
                "recommended_mitigation": self._surface_text(str(advisory.get("recommended_mitigation", ""))),
            },
            "evidence_type": evidence_type,
            "confidence": confidence,
            "verified": relevant,
        }

    def _tool_identity_lookup(self, node: str) -> Dict[str, Any]:
        turn_config = self._turn_config or {}
        strategy = str(turn_config.get("strategy", "benign"))
        target = str(turn_config.get("correct_target", "none"))
        identity = dict(IDENTITY_CONTEXT.get(node, {
            "account": "unknown",
            "account_type": "unknown",
            "privilege_level": "unknown",
            "normal_hosts": [],
        }))

        lookup = {
            "tool": "identity_lookup",
            "turn": self._turn,
            "phase": self._phase,
            "node": node,
            **identity,
            "recent_source_host": node,
            "source_status": "expected",
            "identity_anomaly": False,
            "result_summary": "",
            "confidence": 0.58,
            "evidence_type": "benign",
            "verified": False,
        }

        if node == target and strategy == "lateral_movement":
            lookup.update({
                "recent_source_host": "auth_service",
                "source_status": "unexpected",
                "identity_anomaly": True,
                "confidence": 0.84 if self._task_name != "polymorphic-zero-day" else 0.76,
                "evidence_type": "lateral_movement",
                "verified": True,
            })
        elif node == target and strategy == "supply_chain":
            lookup.update({
                "recent_source_host": "external-release-runner",
                "source_status": "unexpected",
                "identity_anomaly": True,
                "confidence": 0.73,
                "evidence_type": "supply_chain",
                "verified": True,
            })
        elif turn_config.get("is_benign", False):
            lookup.update({
                "recent_source_host": identity.get("normal_hosts", [node])[0] if identity.get("normal_hosts") else node,
                "source_status": "scheduled_change_window",
                "confidence": 0.69,
            })

        if (
            self._task_name == "dual-pivot" and
            strategy == "lateral_movement" and
            self._operational_mode == "evidence_preservation"
        ):
            lookup["source_status"] = "unexpected_but_trackable"
            lookup["result_summary"] = self._surface_text(
                "Identity trail is intact; preserving visibility before hard containment is mission-aligned."
            )
        else:
            lookup["result_summary"] = self._surface_text(
                f"account={lookup['account']} source={lookup['recent_source_host']} anomaly={lookup['identity_anomaly']}"
            )
        return lookup

    def _tool_change_calendar_lookup(self, node: str) -> Dict[str, Any]:
        turn_config = self._turn_config or {}
        strategy = str(turn_config.get("strategy", "benign"))
        target = str(turn_config.get("correct_target", "none"))
        change = dict(CHANGE_CALENDAR.get(node, {
            "window": "none_scheduled",
            "change_type": "none",
            "expected_actor": "unknown",
        }))

        scheduled = bool(turn_config.get("is_benign", False))
        confidence = 0.66 if scheduled else 0.74
        if node == target and strategy == "supply_chain":
            scheduled = False
            confidence = 0.87 if self._task_name != "polymorphic-zero-day" else 0.78
        elif node == target and strategy == "lateral_movement":
            scheduled = False
            confidence = 0.72

        change_status = "scheduled" if scheduled else "no_matching_change"
        if (
            self._task_name == "dual-pivot" and
            strategy == "lateral_movement" and
            self._operational_mode == "evidence_preservation"
        ):
            change_status = "forensic_observation_hold"
        return {
            "tool": "change_calendar_lookup",
            "turn": self._turn,
            "phase": self._phase,
            "node": node,
            **change,
            "scheduled": scheduled,
            "change_status": change_status,
            "confidence": confidence,
            "evidence_type": "benign" if scheduled else ("supply_chain" if node == target and strategy == "supply_chain" else "operational_context"),
            "verified": scheduled or (node == target and strategy == "supply_chain"),
        }

    def _tool_netflow_lookup(self, node: str) -> Dict[str, Any]:
        turn_config = self._turn_config or {}
        strategy = str(turn_config.get("strategy", "benign"))
        target = str(turn_config.get("correct_target", "none"))
        hard_task = self._task_name == "polymorphic-zero-day"

        summary = {
            "tool": "netflow_lookup",
            "turn": self._turn,
            "phase": self._phase,
            "node": node,
            "east_west_connections": 12,
            "outbound_mb": 4,
            "new_destinations": 0,
            "traffic_pattern": "baseline",
            "confidence": 0.57,
            "evidence_type": "benign",
            "verified": False,
        }

        if node == target and strategy == "lateral_movement":
            summary.update({
                "east_west_connections": 46 if hard_task else 58,
                "outbound_mb": 9,
                "new_destinations": 5 if hard_task else 7,
                "traffic_pattern": "east_west_fanout",
                "confidence": 0.79 if hard_task else 0.88,
                "evidence_type": "lateral_movement",
                "verified": True,
            })
        elif node == target and strategy == "exfiltration":
            summary.update({
                "east_west_connections": 18,
                "outbound_mb": 74 if hard_task else 96,
                "new_destinations": 2,
                "traffic_pattern": "outbound_transfer_burst",
                "confidence": 0.82 if hard_task else 0.91,
                "evidence_type": "exfiltration",
                "verified": True,
            })
        elif node == target and strategy == "supply_chain":
            summary.update({
                "east_west_connections": 16,
                "outbound_mb": 19,
                "new_destinations": 1,
                "traffic_pattern": "post_deploy_callback",
                "confidence": 0.71,
                "evidence_type": "supply_chain",
                "verified": True,
            })

        if (
            self._task_name == "polymorphic-zero-day" and
            self._operational_mode in {"forensic_hold", "business_continuity"} and
            node == target and strategy in {"lateral_movement", "exfiltration"}
        ):
            summary["traffic_pattern"] = "controlled_observation_window"
            summary["confidence"] = max(float(summary["confidence"]), 0.83)
        return summary

    def _record_tool_result(self, result: Dict[str, Any]) -> None:
        turn = int(result.get("turn", self._turn) or self._turn)
        internal = {
            "turn": turn,
            "phase": result.get("phase", self._phase),
            "tool": result.get("tool", "unknown"),
            "node": result.get("node", "unknown"),
            "evidence_type": result.get("evidence_type", "unknown"),
            "verified": bool(result.get("verified", False)),
            "confidence": float(result.get("confidence", 0.0) or 0.0),
        }
        self._turn_tool_results.setdefault(turn, []).append(internal)

        trace = {
            "turn": result.get("turn", self._turn),
            "phase": result.get("phase", self._phase),
            "tool": result.get("tool", "unknown"),
            "node": result.get("node", "unknown"),
            "confidence": float(result.get("confidence", 0.0) or 0.0),
            "summary": _tool_summary(result),
        }
        self._tool_trace.append(trace)

        if internal["verified"]:
            self._turn_tool_evidence.setdefault(turn, []).append(internal)

    def _tool_context_for_turn(self) -> Dict[str, Any]:
        evidence = list(self._turn_tool_evidence.get(self._turn, []))
        return {
            "turn": self._turn,
            "tool_count": len([
                row for row in self._tool_trace
                if int(row.get("turn", -1)) == self._turn
            ]),
            "evidence": evidence,
            "tool_results": list(self._turn_tool_results.get(self._turn, [])),
        }

    def _update_foothold_state(
        self,
        p2: Dict[str, str],
        info: Dict[str, Any],
        stage: str,
    ) -> bool:
        if (
            self._task_name != "polymorphic-zero-day" or
            self._foothold_established or
            stage not in ("exploit", "exfiltration")
        ):
            return False

        if p2.get("action") == "monitor" or not info.get("acted_correctly", False):
            self._foothold_established = True
            return True

        return False

    def _register_active_defense(self, p2: Dict[str, str]) -> None:
        action = p2.get("action", "monitor")
        if action not in DEFENSE_TTL:
            return

        target = p2.get("target_node", "unknown")
        self._active_defenses = [
            control for control in self._active_defenses
            if not (control["action"] == action and control["target"] == target)
        ]
        self._active_defenses.append({
            "action": action,
            "target": target,
            "ttl": DEFENSE_TTL[action],
            "side_effect": DEFENSE_SIDE_EFFECT[action],
        })

    def _decay_active_defenses(self) -> None:
        next_controls = []
        for control in self._active_defenses:
            updated = dict(control)
            updated["ttl"] = int(updated.get("ttl", 0)) - 1
            if updated["ttl"] > 0:
                next_controls.append(updated)
        self._active_defenses = next_controls

    def _active_defense_snapshot(self) -> List[Dict[str, Any]]:
        return [dict(control) for control in self._active_defenses]

    def _metadata_with_defenses(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        updated = dict(metadata or {})
        updated["active_defenses"] = self._active_defense_snapshot()
        updated["mission_profile"] = self._mission_profile
        updated["world_split"] = self._world_split
        updated["world_family"] = self._world_family
        updated["operational_mode"] = self._operational_mode
        updated["available_tools"] = [dict(tool) for tool in AVAILABLE_SOC_TOOLS]
        updated["tool_trace"] = [dict(row) for row in self._tool_trace]
        updated["tool_calls_this_turn"] = len([
            row for row in self._tool_trace
            if int(row.get("turn", -1)) == self._turn
        ])
        return updated

    def _with_active_defense_alerts(self, turn_config: Dict[str, Any]) -> Dict[str, Any]:
        if not self._active_defenses:
            return turn_config

        updated = dict(turn_config)
        alerts = list(updated.get("active_alerts", []))
        for control in self._active_defenses:
            alerts.append(
                "[CONTROL] "
                f"{control['action']} active on {control['target']} "
                f"(ttl={control['ttl']}, side_effect={control['side_effect']})"
            )
        updated["active_alerts"] = alerts
        return updated

    def _with_foothold_context(self, turn_config: Dict[str, Any]) -> Dict[str, Any]:
        if (
            self._task_name != "polymorphic-zero-day" or
            not self._foothold_established or
            not turn_config.get("is_benign", False)
        ):
            return turn_config

        updated = dict(turn_config)
        nodes = {name: dict(metrics) for name, metrics in updated.get("network_nodes", {}).items()}
        payment = nodes.setdefault("payment_service", {})
        payment["status"] = "suspicious"
        payment["cpu"] = max(int(payment.get("cpu", 0)), 58)
        payment["request_rate"] = max(int(payment.get("request_rate", 0)), 130)

        alerts = list(updated.get("active_alerts", []))
        alerts.append(
            "[FOOTHOLD] Low-volume internal callback resembles benign service noise, "
            "but prior missed containment means deception is required."
        )

        updated["network_nodes"] = nodes
        updated["active_alerts"] = alerts
        updated["attack_stage"] = self._attacker.current_stage()
        updated["is_benign"] = False
        updated["strategy"] = "lateral_movement"
        updated["correct_action"] = "honeypot"
        updated["correct_target"] = "payment_service"
        updated["foothold_context"] = True
        return updated

    def _prepare_turn_config(self, turn_config: Dict[str, Any]) -> Dict[str, Any]:
        updated = self._with_foothold_context(turn_config)
        updated = self._apply_operational_mode(updated)
        updated = self._with_active_defense_alerts(updated)
        return updated

    def _apply_operational_mode(self, turn_config: Dict[str, Any]) -> Dict[str, Any]:
        updated = dict(turn_config)
        alerts = list(updated.get("active_alerts", []))
        strategy = str(updated.get("strategy", "benign"))
        stage = str(updated.get("attack_stage", "recon"))

        if self._task_name == "dual-pivot" and strategy == "lateral_movement":
            if self._operational_mode == "evidence_preservation":
                updated["correct_action"] = "honeypot"
                alerts.append(
                    self._surface_text(
                        "[MODE] Checkout incident posture favors preserving attacker visibility until identity misuse is corroborated."
                    )
                )
            else:
                updated["correct_action"] = "isolate"
                alerts.append(
                    self._surface_text(
                        "[MODE] Checkout incident posture favors decisive containment once the pivot is corroborated."
                    )
                )

        if self._task_name == "polymorphic-zero-day" and not updated.get("is_benign", False):
            if self._operational_mode in {"forensic_hold", "business_continuity"} and strategy in {"lateral_movement", "exfiltration"}:
                updated["correct_action"] = "honeypot"
                alerts.append(
                    self._surface_text(
                        "[MODE] Enterprise posture favors deception over immediate isolation while mapping the callback path."
                    )
                )
            elif self._operational_mode == "containment_first":
                alerts.append(
                    self._surface_text(
                        "[MODE] Enterprise posture favors immediate containment once compromise is corroborated."
                    )
                )
            if self._operational_mode == "business_continuity" and stage == "exploit":
                nodes = {name: dict(metrics) for name, metrics in updated.get("network_nodes", {}).items()}
                payment = nodes.setdefault("payment_service", {})
                payment["status"] = "elevated"
                payment["request_rate"] = max(int(payment.get("request_rate", 0)), 122)
                updated["network_nodes"] = nodes
                alerts.append(
                    self._surface_text(
                        "[MODE] Customer traffic remains sensitive; service continuity pressure is elevated during this window."
                    )
                )

        updated["active_alerts"] = alerts
        updated["world_split"] = self._world_split
        updated["world_family"] = self._world_family
        updated["operational_mode"] = self._operational_mode
        return updated

    def _surface_text(self, text: str) -> str:
        return self._attacker._surface(text)


def _action_value(value: Any, default: str) -> str:
    """Serialize action fields without leaking Enum member names."""
    if value is None:
        return default
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _action_float(value: Any, default: float) -> float:
    """Coerce optional numeric action fields to floats with a safe fallback."""
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _replay_result(info: Dict[str, Any]) -> str:
    """Map grader text into compact replay result labels."""
    reason = str(info.get("score_reason", "")).lower()
    if "false positive" in reason:
        return "false_positive"
    if reason.startswith("unverified"):
        return "unverified"
    if reason.startswith("optimal") or reason.startswith("correct") or reason.startswith("context-aware optimal"):
        return "optimal"
    if reason.startswith("heavy-handed"):
        return "heavy"
    return "wrong"


def _tool_summary(result: Dict[str, Any]) -> str:
    if result.get("error"):
        return str(result["error"])[:120]
    if result.get("tool") == "log_search":
        events = result.get("events") or []
        return str(events[0])[:120] if events else "no matching log events"
    if result.get("tool") == "cmdb_lookup":
        deps = result.get("dependencies") or []
        return f"{result.get('node')} criticality={result.get('criticality')} deps={len(deps)}"
    if result.get("tool") == "edr_status":
        return (
            f"containment={result.get('containment')} "
            f"beaconing={result.get('beaconing')} "
            f"persistence={result.get('persistence')}"
        )
    if result.get("tool") == "vuln_lookup":
        return f"risk={result.get('risk')} finding={result.get('finding')}"
    if result.get("tool") == "identity_lookup":
        return (
            f"account={result.get('account')} "
            f"source={result.get('recent_source_host')} "
            f"anomaly={result.get('identity_anomaly')}"
        )
    if result.get("tool") == "change_calendar_lookup":
        return (
            f"scheduled={result.get('scheduled')} "
            f"window={result.get('window')} "
            f"change={result.get('change_type')}"
        )
    if result.get("tool") == "netflow_lookup":
        return (
            f"pattern={result.get('traffic_pattern')} "
            f"east_west={result.get('east_west_connections')} "
            f"outbound_mb={result.get('outbound_mb')}"
        )
    return str(result)[:120]


def _public_tool_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Return observable tool facts without leaking grader labels."""
    hidden = {"evidence_type", "verified"}
    public = {key: value for key, value in result.items() if key not in hidden}
    public["result_summary"] = _tool_summary(result)
    return public


def _degrade_handoff(
    phase1_output: Dict[str, Any],
    turn_config: Dict[str, Any],
    task_name: str,
    turn: int,
) -> Dict[str, Any]:
    """Degrade hard-task handoffs after the strategy shift without changing grading."""
    output = dict(phase1_output)
    if (
        task_name == "dual-pivot" and
        turn >= 4 and
        turn_config.get("strategy") == "lateral_movement"
    ):
        output["confidence"] = min(float(output.get("confidence", 0.5)), 0.56)
        output["recommended_action"] = "monitor"
        output["handoff_quality"] = "degraded"
        output["handoff_note"] = (
            "Analyst handoff is less reliable after the pivot; executor should verify "
            "internal movement before taking payment_service offline."
        )
    elif (
        task_name == "polymorphic-zero-day" and
        turn >= 4 and
        turn_config.get("strategy") in {"lateral_movement", "exfiltration", "supply_chain"}
    ):
        output["confidence"] = min(float(output.get("confidence", 0.5)), 0.36)
        output["recommended_action"] = "monitor"
        output["handoff_quality"] = "degraded"
        output["handoff_note"] = (
            "Analyst confidence degraded after attacker pivot; executor must verify "
            "whether monitor is too passive for the current stage."
        )
    else:
        output["handoff_quality"] = "clean"
    return output
