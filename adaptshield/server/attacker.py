"""
AdaptShield Attacker Engine

Deterministic scripted Python adversary — NEVER an LLM.
Implements attack stage progression within episodes.
Rotates strategies between episodes on a fixed schedule.

The internal episode counter is NEVER passed to the agent.
Agent must infer strategy from network signals only.
"""

import os
import random
from typing import Any, Dict, Tuple


# ── Strategy definitions ───────────────────────────────────────────────────
STRATEGY_MAP: Dict[str, Dict[str, str]] = {
    "brute_force": {
        "correct_action": "rate_limit",
        "correct_target": "auth_service",
        "description":    "Credential stuffing attack on authentication service",
    },
    "lateral_movement": {
        "correct_action": "isolate",
        "correct_target": "payment_service",
        "description":    "Attacker pivoting from compromised host to internal nodes",
    },
    "exfiltration": {
        "correct_action": "honeypot",
        "correct_target": "database",
        "description":    "Slow sustained data exfiltration from database",
    },
    "supply_chain": {
        "correct_action": "patch",
        "correct_target": "api_gateway",
        "description":    "Malicious package injected into update pipeline",
    },
}

# Strategy rotation order
STRATEGY_ROTATION = [
    "brute_force",
    "lateral_movement",
    "exfiltration",
    "supply_chain",
]

# Attack stages within an episode
# Each stage escalates if agent fails to act
ATTACK_STAGES = ["recon", "exploit", "exfiltration"]

# Baseline healthy network metrics
BASELINE_NODES: Dict[str, Dict[str, Any]] = {
    "auth_service":    {"status": "healthy", "request_rate": 120, "error_rate": 0.01, "cpu": 22, "outbound_mb": 2},
    "payment_service": {"status": "healthy", "request_rate": 85,  "error_rate": 0.01, "cpu": 31, "outbound_mb": 5},
    "database":        {"status": "healthy", "request_rate": 34,  "error_rate": 0.01, "cpu": 18, "outbound_mb": 8},
    "api_gateway":     {"status": "healthy", "request_rate": 203, "error_rate": 0.02, "cpu": 29, "outbound_mb": 3},
}

SOURCE_SUBNETS = ["192.168.1.x", "10.44.8.x", "172.16.22.x", "203.0.113.x"]
SERVICE_ACCOUNTS = ["svc_internal", "svc_billing", "svc_reporter", "deploy_bot"]
PACKAGE_NAMES = ["core-auth-lib", "gateway-router", "payment-sdk", "session-cache"]
DB_TABLES = ["customer_tokens", "invoice_archive", "payment_methods", "audit_events"]
ALERT_SOURCES = ["SIEM", "EDR", "WAF", "NETFLOW"]


class AttackerEngine:
    """
    Polymorphic scripted attacker with stage progression.

    Within an episode: attack progresses through recon → exploit → exfiltration
    if the agent fails to act correctly. Early correct action stops progression.

    Between episodes: strategy rotates on a fixed schedule per task.
    Hard task additionally shifts strategy mid-episode after turn 3.
    """

    def __init__(self, task_name: str):
        random.seed(int(os.environ.get("ADAPTSHIELD_SEED", random.randint(0, 9999))))

        self.task_name     = task_name
        self._episode      = 0   # internal — NEVER passed to agent
        self._turn         = 0   # within-episode turn counter
        self._stage_idx    = 0   # current attack stage index
        self._escalated    = False  # did agent miss a turn?

        self._shift_every = {
            "direct-triage":        9999,
            "dual-pivot":           20,
            "polymorphic-zero-day": 10,
        }.get(task_name, 9999)

        self._noise_rate = 0.15 if task_name == "polymorphic-zero-day" else 0.0

    # ── Public interface ───────────────────────────────────────────────────

    def reset_episode(self) -> None:
        """Called at the start of each episode."""
        self._turn      = 0
        self._stage_idx = 0
        self._escalated = False

    def current_strategy(self) -> str:
        """Return active strategy — accounts for mid-episode shift on hard task."""
        base_idx = (self._episode // self._shift_every) % len(STRATEGY_ROTATION)

        # Medium task: one clean pivot from brute force to lateral movement.
        if self.task_name == "dual-pivot" and self._turn >= 3:
            base_idx = (base_idx + 1) % 2

        # Hard task: strategy shifts mid-episode after turn 3
        if self.task_name == "polymorphic-zero-day" and self._turn >= 3:
            base_idx = (base_idx + 1) % len(STRATEGY_ROTATION)

        return STRATEGY_ROTATION[base_idx]

    def current_stage(self) -> str:
        """Return current attack stage."""
        return ATTACK_STAGES[min(self._stage_idx, len(ATTACK_STAGES) - 1)]

    def build_observation(self) -> Dict[str, Any]:
        """
        Build the network observation for current turn.
        Returns ambiguous signals — no explicit strategy names or codes.
        Agent must reason about what the metrics mean.
        """
        is_benign = random.random() < self._noise_rate
        strategy  = self.current_strategy()
        stage     = self.current_stage()

        nodes  = {k: dict(v) for k, v in BASELINE_NODES.items()}
        alerts = []

        if is_benign:
            alerts = self._build_benign_alerts()
            nodes["api_gateway"]["cpu"]    = 71
            nodes["api_gateway"]["status"] = "elevated"
            return {
                "network_nodes":  nodes,
                "active_alerts":  alerts,
                "attack_stage":   "none",
                "is_benign":      True,
                "strategy":       "benign",
                "correct_action": "monitor",
                "correct_target": "none",
            }

        nodes, alerts = self._build_attack_state(strategy, stage, nodes)

        return {
            "network_nodes":  nodes,
            "active_alerts":  alerts,
            "attack_stage":   stage,
            "is_benign":      False,
            "strategy":       strategy,
            "correct_action": STRATEGY_MAP[strategy]["correct_action"],
            "correct_target": STRATEGY_MAP[strategy]["correct_target"],
        }

    def advance_turn(self, agent_acted_correctly: bool) -> None:
        """
        Advance turn counter. Escalate attack stage if agent failed.
        Called after grading each Phase 2 action.
        """
        self._turn += 1
        if not agent_acted_correctly:
            self._escalated = True
            self._stage_idx = min(self._stage_idx + 1, len(ATTACK_STAGES) - 1)

    def advance_episode(self) -> None:
        """Called at the end of each episode."""
        self._episode += 1
        self.reset_episode()

    # ── Internal signal builders ───────────────────────────────────────────

    def _build_benign_alerts(self) -> list:
        """Build convincing but harmless noise alerts."""
        deploy_user = random.choice(SERVICE_ACCOUNTS)
        alert_prefix = self._alert_prefix()
        options = [
            [
                f"{alert_prefix} Scheduled database backup completed: 2.3GB archived",
                f"{alert_prefix} User '{deploy_user}' executed routine deployment script",
                f"{alert_prefix} api_gateway: elevated CPU due to analytics batch job",
            ],
            [
                f"{alert_prefix} Certificate renewal completed for *.internal.corp",
                f"{alert_prefix} Routine log rotation completed on all nodes",
                f"{alert_prefix} payment_service: normal traffic spike during billing window",
            ],
            [
                f"{alert_prefix} Health check: all nodes nominal",
                f"{alert_prefix} auth_service: password policy enforcement triggered for 3 users",
                f"{alert_prefix} Scheduled vulnerability scan completed: 0 critical findings",
            ],
        ]
        return random.choice(options)

    def _build_attack_state(
        self,
        strategy: str,
        stage: str,
        nodes: Dict,
    ) -> Tuple[Dict, list]:
        """
        Build network state and alerts for an active attack.
        Signals are intentionally ambiguous — no strategy names or MITRE codes.
        Severity escalates with attack stage.
        """
        stage_multiplier = {"recon": 1.0, "exploit": 1.6, "exfiltration": 2.4}
        mult = stage_multiplier.get(stage, 1.0)

        alerts = []
        alert_prefix = self._alert_prefix()

        if strategy == "brute_force":
            subnet = random.choice(SOURCE_SUBNETS)
            nodes["auth_service"]["status"]       = "degraded" if stage != "recon" else "elevated"
            nodes["auth_service"]["request_rate"] = int(340 * mult)
            nodes["auth_service"]["error_rate"]   = min(0.48, 0.18 * mult)
            nodes["auth_service"]["cpu"]          = min(96, int(45 * mult))
            alerts = [
                f"{alert_prefix} auth_service: {int(120 * mult)} failed login attempts in last 60s from subnet {subnet}",
                f"{alert_prefix} auth_service: error rate {nodes['auth_service']['error_rate']:.0%}; {stage} activity detected",
                f"{alert_prefix} payment_service: normal transaction volume",
            ]

        elif strategy == "lateral_movement":
            account = random.choice(SERVICE_ACCOUNTS)
            nodes["payment_service"]["status"]       = "suspicious"
            nodes["payment_service"]["request_rate"] = int(95 * mult)
            nodes["payment_service"]["cpu"]          = int(38 * mult)
            nodes["auth_service"]["request_rate"]    = int(155 * mult)
            add_auth_distractor = (
                (self.task_name == "dual-pivot" and self._turn == 3) or
                (self.task_name == "polymorphic-zero-day" and self._turn >= 3)
            )
            if add_auth_distractor:
                nodes["auth_service"]["status"]     = "elevated"
                nodes["auth_service"]["error_rate"] = min(0.36, 0.14 * mult)
                nodes["auth_service"]["cpu"]        = min(90, int(42 * mult))
            alerts = [
                f"{alert_prefix} Unusual internal connections: auth_service -> payment_service ({int(40 * mult)}/min)",
                f"{alert_prefix} payment_service: service account '{account}' authenticated from new host",
                f"{alert_prefix} Cross-node traffic volume {mult:.1f}x above 7-day baseline",
            ]
            if add_auth_distractor:
                alerts.append(
                    f"{alert_prefix} auth_service: residual failed-login noise from earlier credential attack"
                )

        elif strategy == "exfiltration":
            table = random.choice(DB_TABLES)
            nodes["database"]["status"]            = "suspicious" if stage != "recon" else "elevated"
            nodes["database"]["outbound_mb"]       = int(8 * mult * 12)
            nodes["database"]["request_rate"]      = int(34 * mult * 3)
            nodes["database"]["cpu"]               = int(18 * mult * 1.5)
            alerts = [
                f"{alert_prefix} database: outbound traffic {nodes['database']['outbound_mb']}MB/hr; {mult:.1f}x above baseline",
                f"{alert_prefix} database: {int(280 * mult)} sequential reads from table '{table}'",
                f"{alert_prefix} Large compressed archive creation activity on database node",
            ]

        elif strategy == "supply_chain":
            package = random.choice(PACKAGE_NAMES)
            nodes["api_gateway"]["status"] = "suspicious"
            nodes["api_gateway"]["cpu"]    = int(29 + 20 * mult)
            alerts = [
                f"{alert_prefix} api_gateway: unsigned package update request received from external source",
                f"{alert_prefix} api_gateway: binary hash mismatch on dependency '{package}'",
                f"{alert_prefix} Unexpected outbound connection from api_gateway to unrecognized host",
            ]

        return nodes, alerts

    def _alert_prefix(self) -> str:
        """Return deterministic-looking SOC alert metadata under ADAPTSHIELD_SEED."""
        source = random.choice(ALERT_SOURCES)
        alert_id = random.randint(1000, 9999)
        minute = random.randint(0, 59)
        second = random.randint(0, 59)
        return f"[{source}-{alert_id} 03:{minute:02d}:{second:02d}Z]"
