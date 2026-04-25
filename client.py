# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""AdaptShield environment client."""

from typing import Any, Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from models import AdaptShieldAction, AdaptShieldObservation


class AdaptshieldEnv(
    EnvClient[AdaptShieldAction, AdaptShieldObservation, State]
):
    """
    Client for the Adaptshield Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with AdaptshieldEnv(base_url="http://localhost:7860") as client:
        ...     result = client.reset()
        ...     print(result.observation.phase)
        ...
        ...     result = client.step(AdaptShieldAction(
        ...         threat_type="brute_force",
        ...         confidence=0.9,
        ...         target_node="auth_service",
        ...         recommended_action="rate_limit",
        ...     ))
        ...     print(result.observation.phase1_assessment)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = AdaptshieldEnv.from_docker_image("adaptshield-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(AdaptShieldAction(
        ...         threat_type="benign",
        ...         confidence=0.8,
        ...         target_node="auth_service",
        ...         recommended_action="monitor",
        ...     ))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: AdaptShieldAction) -> Dict[str, Any]:
        """
        Convert AdaptShieldAction to a JSON-safe payload.

        Args:
            action: AdaptShieldAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump(
            mode="json",
            exclude_none=True,
            exclude_defaults=True,
        )

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[AdaptShieldObservation]:
        """
        Parse server response into StepResult[AdaptShieldObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with AdaptShieldObservation
        """
        obs_data = dict(payload.get("observation", {}))
        obs_data.setdefault("done", payload.get("done", False))
        obs_data.setdefault("reward", payload.get("reward", 0.0))
        observation = AdaptShieldObservation(**obs_data)

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
