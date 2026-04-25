#!/usr/bin/env python3
"""Shared SOC investigation helpers for AdaptShield agents."""

from __future__ import annotations

import json
import urllib.request
from typing import Any, Callable, Dict, List, Optional


THREAT_TOOL_PLAN = {
    "brute_force": [("log_search", "auth_service"), ("cmdb_lookup", "auth_service")],
    "lateral_movement": [("edr_status", "payment_service"), ("log_search", "payment_service")],
    "exfiltration": [("log_search", "database"), ("edr_status", "database")],
    "supply_chain": [("vuln_lookup", "api_gateway"), ("log_search", "api_gateway")],
    "benign": [("cmdb_lookup", "api_gateway")],
}

FALLBACK_SWEEP = [
    ("edr_status", "payment_service"),
    ("log_search", "database"),
    ("vuln_lookup", "api_gateway"),
]


def classify_from_metrics(network_nodes: Dict[str, Dict[str, Any]]) -> str:
    auth = network_nodes.get("auth_service", {})
    payment = network_nodes.get("payment_service", {})
    database = network_nodes.get("database", {})
    gateway = network_nodes.get("api_gateway", {})

    if float(auth.get("error_rate", 0.0)) >= 0.10:
        return "brute_force"
    if payment.get("status") == "suspicious" or float(payment.get("cpu", 0)) >= 55:
        return "lateral_movement"
    if float(database.get("outbound_mb", 0)) >= 50:
        return "exfiltration"
    if gateway.get("status") == "suspicious":
        return "supply_chain"
    return "benign"


def investigate_local(env: Any, obs: Any, use_tools: bool) -> List[Dict[str, Any]]:
    """Query local environment tool methods before Phase 1 action."""
    return investigate_local_with_depth(env, obs, use_tools=use_tools, thorough=False)


def investigate_local_with_depth(
    env: Any,
    obs: Any,
    use_tools: bool,
    thorough: bool,
) -> List[Dict[str, Any]]:
    """Query local tools; thorough mode adds evidence-fusion follow-ups."""
    if not use_tools or getattr(obs, "phase", 1) != 1:
        return []
    task_name = getattr(obs, "task_name", "")
    if task_name == "direct-triage":
        return []

    threat = classify_from_metrics(getattr(obs, "network_nodes", {}))
    if task_name == "dual-pivot":
        tool_name, node = THREAT_TOOL_PLAN.get(threat, THREAT_TOOL_PLAN["benign"])[0]
        return [env.call_tool(tool_name, node=node)]

    if task_name != "polymorphic-zero-day":
        return []

    results = []
    for tool_name, node in THREAT_TOOL_PLAN.get(threat, THREAT_TOOL_PLAN["benign"]):
        results.append(env.call_tool(tool_name, node=node))

    if not has_attack_indicators(results):
        for tool_name, node in FALLBACK_SWEEP:
            if (tool_name, node) not in THREAT_TOOL_PLAN.get(threat, []):
                results.append(env.call_tool(tool_name, node=node))
    if thorough:
        _complete_evidence_fusion(
            call_tool=lambda tool_name, node: env.call_tool(tool_name, node=node),
            results=results,
        )
    return results


def investigate_http(
    env_base_url: str,
    session_id: Optional[str],
    obs: Dict[str, Any],
    use_tools: bool,
    thorough: bool = False,
) -> List[Dict[str, Any]]:
    """Query SOC HTTP tool endpoints for a persistent /soc session."""
    if not use_tools or not session_id or int(obs.get("phase", 1)) != 1:
        return []
    task_name = obs.get("task_name")
    if task_name == "direct-triage":
        return []

    threat = classify_from_metrics(obs.get("network_nodes", {}))
    results: List[Dict[str, Any]] = []

    def call(tool_name: str, node: str) -> Dict[str, Any]:
        path = f"/tools/{tool_name}"
        payload = {"session_id": session_id, "node": node}
        return http_post(env_base_url, path, payload)

    if task_name == "dual-pivot":
        tool_name, node = THREAT_TOOL_PLAN.get(threat, THREAT_TOOL_PLAN["benign"])[0]
        return [call(tool_name, node)]

    if task_name != "polymorphic-zero-day":
        return []

    for tool_name, node in THREAT_TOOL_PLAN.get(threat, THREAT_TOOL_PLAN["benign"]):
        results.append(call(tool_name, node))

    if not has_attack_indicators(results):
        for tool_name, node in FALLBACK_SWEEP:
            if (tool_name, node) not in THREAT_TOOL_PLAN.get(threat, []):
                results.append(call(tool_name, node))
    if thorough:
        _complete_evidence_fusion(call_tool=call, results=results)
    return results


def has_attack_indicators(results: List[Dict[str, Any]]) -> bool:
    return infer_threat_from_tool_results(results)["threat_type"] != "benign"


def infer_threat_from_tool_results(results: List[Dict[str, Any]]) -> Dict[str, str]:
    """Infer threat from observable tool facts only, not grader labels."""
    text_blob = " ".join(
        [
            str(result.get("result_summary", "")) + " " +
            " ".join(str(event) for event in result.get("events", [])) + " " +
            str(result.get("finding", "")) + " " +
            str(result.get("process_note", ""))
            for result in results
        ]
    ).lower()

    if any(result.get("risk") == "critical" for result in results) or "hash mismatch" in text_blob:
        return {"threat_type": "supply_chain", "target_node": "api_gateway", "action": "patch"}
    if "sequential reads" in text_blob or "compressed archive" in text_blob or "egress exceeds" in text_blob:
        return {"threat_type": "exfiltration", "target_node": "database", "action": "honeypot"}
    if any(result.get("beaconing") for result in results) or "service account" in text_blob or "internal sessions" in text_blob:
        return {"threat_type": "lateral_movement", "target_node": "payment_service", "action": "isolate"}
    if "failed logins" in text_blob or "password spray" in text_blob:
        return {"threat_type": "brute_force", "target_node": "auth_service", "action": "rate_limit"}
    return {"threat_type": "benign", "target_node": "api_gateway", "action": "monitor"}


def _complete_evidence_fusion(
    call_tool: Callable[[str, str], Dict[str, Any]],
    results: List[Dict[str, Any]],
) -> None:
    belief = infer_threat_from_tool_results(results)
    threat = belief["threat_type"]
    if threat == "benign":
        return

    called = {
        (str(result.get("tool", "")), str(result.get("node", "")))
        for result in results
    }
    for tool_name, node in THREAT_TOOL_PLAN.get(threat, []):
        if (tool_name, node) not in called:
            results.append(call_tool(tool_name, node))


def attach_tool_results(obs: Dict[str, Any], tool_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    updated = dict(obs)
    updated["tool_results"] = tool_results
    return updated


def summarize_tool_results(tool_results: List[Dict[str, Any]]) -> str:
    if not tool_results:
        return "No SOC tools queried for this turn."

    lines = []
    for result in tool_results:
        lines.append(json.dumps(_compact_result(result), separators=(",", ":")))
    return "\n".join(lines)


def http_post(env_base_url: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{env_base_url.rstrip('/')}{path}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=60) as response:
        return json.loads(response.read())


def _compact_result(result: Dict[str, Any]) -> Dict[str, Any]:
    keep = [
        "tool",
        "node",
        "evidence_type",
        "verified",
        "confidence",
        "events",
        "containment",
        "persistence",
        "beaconing",
        "criticality",
        "dependencies",
        "risk",
        "finding",
        "recommended_mitigation",
        "safe_actions",
    ]
    return {key: result[key] for key in keep if key in result}
