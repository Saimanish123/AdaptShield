"""
AdaptShield FastAPI Server

CRITICAL: Uses factory pattern (make_env function), NOT singleton.
Singleton was the Round 1 failure — always served wrong task.
Factory creates a fresh isolated instance per evaluator session.

openenv validate requires:
- def main() function present
- called as main() in if __name__ block (literal string check)
- port 7860 (HF Spaces default)
"""

import os
import sys
from typing import Any, Dict
from uuid import uuid4

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from fastapi import Body, HTTPException
    from openenv.core.env_server.http_server import create_app
except Exception as e:
    raise ImportError(
        "openenv-core required. Install: pip install openenv-core"
    ) from e

from models import AdaptShieldAction, AdaptShieldObservation
from server.adaptshield_environment import AdaptShieldEnvironment

DEFAULT_TASK = os.getenv("ADAPTSHIELD_TASK", "direct-triage")
SOC_SESSIONS: Dict[str, AdaptShieldEnvironment] = {}


def make_env() -> AdaptShieldEnvironment:
    """
    Factory function — fresh isolated instance per session.
    Never a singleton. Evaluator sessions must be independent.
    """
    return AdaptShieldEnvironment(task_name=DEFAULT_TASK)


app = create_app(
    make_env,
    AdaptShieldAction,
    AdaptShieldObservation,
    env_name="adaptshield",
    max_concurrent_envs=10,
)


@app.post("/soc/reset", tags=["AdaptShield SOC Tools"])
async def soc_reset(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Start a persistent demo session for SOC tool/API workflows."""
    task = str(payload.get("task", DEFAULT_TASK))
    env = AdaptShieldEnvironment(task_name=task)
    obs = env.reset()
    session_id = str(uuid4())
    SOC_SESSIONS[session_id] = env
    return {
        "session_id": session_id,
        "observation": obs.model_dump(mode="json"),
        "available_tools": obs.metadata.get("available_tools", []),
    }


@app.post("/soc/step", tags=["AdaptShield SOC Tools"])
async def soc_step(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Step a persistent SOC tool/API session."""
    env = _soc_session(payload)
    try:
        action = AdaptShieldAction(**dict(payload.get("action", {})))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    obs = env.step(action)
    return {
        "session_id": payload.get("session_id"),
        "observation": obs.model_dump(mode="json"),
        "reward": float(obs.reward),
        "done": bool(obs.done),
    }


@app.post("/tools/log_search", tags=["AdaptShield SOC Tools"])
async def tool_log_search(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Search stateful SIEM/application logs for the active session."""
    return _soc_session(payload).call_tool(
        "log_search",
        node=payload.get("node", payload.get("target_node", "unknown")),
        query=payload.get("query", ""),
    )


@app.post("/tools/cmdb_lookup", tags=["AdaptShield SOC Tools"])
async def tool_cmdb_lookup(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Look up service ownership, criticality, and dependency blast radius."""
    return _soc_session(payload).call_tool(
        "cmdb_lookup",
        node=payload.get("node", payload.get("target_node", "unknown")),
    )


@app.post("/tools/edr_status", tags=["AdaptShield SOC Tools"])
async def tool_edr_status(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Check endpoint containment and persistence indicators."""
    return _soc_session(payload).call_tool(
        "edr_status",
        node=payload.get("node", payload.get("target_node", "unknown")),
    )


@app.post("/tools/vuln_lookup", tags=["AdaptShield SOC Tools"])
async def tool_vuln_lookup(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Query internal vulnerability/advisory evidence for a service package."""
    return _soc_session(payload).call_tool(
        "vuln_lookup",
        node=payload.get("node", payload.get("target_node", "unknown")),
        package=payload.get("package", ""),
    )


@app.post("/tools/identity_lookup", tags=["AdaptShield SOC Tools"])
async def tool_identity_lookup(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Inspect account behavior and unusual source-host affinity for a service identity."""
    return _soc_session(payload).call_tool(
        "identity_lookup",
        node=payload.get("node", payload.get("target_node", "unknown")),
    )


@app.post("/tools/change_calendar_lookup", tags=["AdaptShield SOC Tools"])
async def tool_change_calendar_lookup(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Check whether a deploy or maintenance window was actually scheduled."""
    return _soc_session(payload).call_tool(
        "change_calendar_lookup",
        node=payload.get("node", payload.get("target_node", "unknown")),
    )


@app.post("/tools/netflow_lookup", tags=["AdaptShield SOC Tools"])
async def tool_netflow_lookup(payload: Dict[str, Any] = Body(default_factory=dict)) -> Dict[str, Any]:
    """Inspect east-west and outbound traffic summaries for the active session."""
    return _soc_session(payload).call_tool(
        "netflow_lookup",
        node=payload.get("node", payload.get("target_node", "unknown")),
    )


def _soc_session(payload: Dict[str, Any]) -> AdaptShieldEnvironment:
    session_id = str(payload.get("session_id", ""))
    env = SOC_SESSIONS.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail="Unknown SOC session. Call /soc/reset first.",
        )
    return env


def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    """Start the uvicorn server. Call main() to run."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    main()
