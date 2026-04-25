# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# AdaptShield — Pydantic Data Models
#
# CRITICAL DESIGN DECISION: Phase1Action and Phase2Action are SEPARATE classes.
# A single combined class with optional fields causes 500 errors when the
# evaluator sends a Phase 2 payload and Pydantic tries to validate Phase 1 fields.

from enum import Enum
from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field, model_validator


class DefenseAction(str, Enum):
    """
    Strict action space for the Tactical Executor (Phase 2).
    Using Enum prevents LLM hallucination from reaching the grader.
    """
    RATE_LIMIT = "rate_limit"  # Light — throttles traffic, keeps service online
    ISOLATE    = "isolate"     # Heavy — takes node offline, stops spread
    HONEYPOT   = "honeypot"    # Strategic — redirects attacker to decoy
    PATCH      = "patch"       # Targeted — fixes supply chain vulnerability
    MONITOR    = "monitor"     # Passive — gather info, risk escalation


class ThreatType(str, Enum):
    """Known attack strategies the Threat Analyst can classify."""
    BRUTE_FORCE      = "brute_force"
    LATERAL_MOVEMENT = "lateral_movement"
    EXFILTRATION     = "exfiltration"
    SUPPLY_CHAIN     = "supply_chain"
    BENIGN           = "benign"


class Phase1Action(Action):
    """
    Threat Analyst output — pure reasoning, no defensive action.

    The agent reads raw network state and produces a structured
    threat assessment. This is graded independently for classification
    accuracy before Phase 2 acts on it.
    """
    threat_type:        str             = Field(
        ...,
        description="Identified attack strategy: brute_force, lateral_movement, "
                    "exfiltration, supply_chain, or benign",
    )
    confidence:         float           = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in the threat classification (0.0 to 1.0)",
    )
    target_node:        str             = Field(
        ...,
        description="Primary affected node: auth_service, payment_service, "
                    "database, or api_gateway",
    )
    recommended_action: DefenseAction   = Field(
        ...,
        description="Recommended defense action for Phase 2 to execute",
    )
    reasoning:          Optional[str]   = Field(
        default=None,
        description="Chain of thought. Not graded. Helps training stability.",
    )


class Phase2Action(Action):
    """
    Tactical Executor output — defensive action based ONLY on Phase 1 assessment.

    Phase 2 agent is deliberately blind to raw network state.
    It receives only the Phase 1 threat assessment and must act on it.
    """
    action:      DefenseAction  = Field(
        ...,
        description="Defense action to execute",
    )
    target_node: str            = Field(
        ...,
        description="Node to apply action to: auth_service, payment_service, "
                    "database, or api_gateway",
    )
    reasoning:   Optional[str]  = Field(
        default=None,
        description="Chain of thought. Not graded.",
    )


class AdaptShieldAction(Action):
    """
    Unified action model accepted by the OpenEnv HTTP server.

    The environment alternates between two phases, so the transport layer must
    accept either a Threat Analyst payload or a Tactical Executor payload.
    Validation keeps those shapes distinct while still fitting the single
    action model expected by `create_app`.
    """

    threat_type: Optional[str] = Field(
        default=None,
        description="Phase 1 only: identified attack strategy",
    )
    confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Phase 1 only: confidence in the threat classification",
    )
    target_node: Optional[str] = Field(
        default=None,
        description="Target node for either phase",
    )
    recommended_action: Optional[DefenseAction] = Field(
        default=None,
        description="Phase 1 only: recommended follow-up action",
    )
    action: Optional[DefenseAction] = Field(
        default=None,
        description="Phase 2 only: defensive action to execute",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional one-sentence rationale",
    )

    @model_validator(mode="after")
    def validate_phase_shape(self) -> "AdaptShieldAction":
        phase1_present = any(
            value is not None
            for value in (self.threat_type, self.confidence, self.recommended_action)
        )
        phase2_present = self.action is not None

        if phase1_present and phase2_present:
            raise ValueError(
                "Action payload must be either Phase 1 or Phase 2, not both."
            )
        if not phase1_present and not phase2_present:
            raise ValueError(
                "Action payload must contain Phase 1 fields or a Phase 2 action."
            )

        if phase1_present:
            missing = [
                field_name
                for field_name, value in (
                    ("threat_type", self.threat_type),
                    ("confidence", self.confidence),
                    ("target_node", self.target_node),
                    ("recommended_action", self.recommended_action),
                )
                if value is None
            ]
        else:
            missing = [
                field_name
                for field_name, value in (
                    ("action", self.action),
                    ("target_node", self.target_node),
                )
                if value is None
            ]

        if missing:
            raise ValueError(
                f"Missing required fields for this phase: {', '.join(missing)}"
            )

        return self


class AdaptShieldObservation(Observation):
    """
    Observation returned after each step.

    Phase 1 observation: contains full network state (network_nodes, active_alerts).
    Phase 2 observation: network_nodes and active_alerts are EMPTY.
                         phase1_assessment contains the Phase 1 output.

    Episode number is NEVER included — agent must rely on signals only.
    """

    # Identity
    scenario_id:    str             = Field(default="")
    task_name:      str             = Field(default="")
    phase:          int             = Field(default=1,
        description="1 = Threat Analyst turn, 2 = Tactical Executor turn")
    turn:           int             = Field(default=0)
    max_turns:      int             = Field(default=5)

    # Network state — populated in Phase 1, EMPTY in Phase 2
    network_nodes:  Dict[str, Any]  = Field(default_factory=dict)
    active_alerts:  List[str]       = Field(default_factory=list)
    attack_stage:   str             = Field(
        default="none",
        description="Current attack progression stage: recon, exploit, exfiltration, none",
    )

    # Rolling history of last 3 turns
    history:        List[Dict[str, str]] = Field(default_factory=list)

    # Phase 2 only — Phase 1 output passed to executor
    phase1_assessment: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Populated only in Phase 2. Phase 2 agent sees ONLY this.",
    )

    # Context
    system_context:    str          = Field(default="")
    available_actions: List[str]    = Field(default_factory=list)

    # Feedback
    last_action_result: Optional[str] = Field(default=None)
    reward:             float          = Field(default=0.0)
    done:               bool           = Field(default=False)
    metadata:           Dict[str, Any] = Field(default_factory=dict)

    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Keep metadata in OpenEnv HTTP observation payloads.

        OpenEnv's serializer excludes metadata from the nested observation by
        default. AdaptShield exposes normalized_score there, so we remove only
        that exclusion while preserving the serializer's reward/done handling.
        """
        exclude = kwargs.get("exclude")
        if isinstance(exclude, set) and "metadata" in exclude:
            kwargs["exclude"] = set(exclude) - {"metadata"}
        elif isinstance(exclude, dict) and "metadata" in exclude:
            kwargs["exclude"] = {
                key: value for key, value in exclude.items() if key != "metadata"
            }
        return super().model_dump(*args, **kwargs)


# Backward-compatible aliases for earlier package names.
AdaptshieldAction = AdaptShieldAction
AdaptshieldObservation = AdaptShieldObservation
