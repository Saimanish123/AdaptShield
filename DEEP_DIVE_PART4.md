# AdaptShield Deep Dive — Part 4: Environment, SOC Tools & Server

---

## PART 6 — The Environment Class (`adaptshield_environment.py`)

### 6.1 Class Overview

`AdaptShieldEnvironment` (967 lines) is the central orchestrator. It inherits from OpenEnv's `Environment` base and implements the full two-phase episode lifecycle. Every concern — phase transitions, attacker state, active defenses, tool evidence, foothold tracking, history, episode replay — lives here.

```python
class AdaptShieldEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
```

The `SUPPORTS_CONCURRENT_SESSIONS = True` flag tells the OpenEnv framework this environment can handle multiple simultaneous evaluator sessions without shared state. This works because `make_env()` in `app.py` creates a **fresh isolated instance per session** — there is no global singleton.

---

### 6.2 Constructor State

```python
def __init__(self, task_name: str = "direct-triage"):
    self._task_name             # which of the 3 tasks
    self._config                # task config dict (max_turns, mission_profile)
    self._mission_profile       # copy of mission_profile for this episode
    self._attacker              # AttackerEngine instance
    self._state                 # OpenEnv State (episode_id UUID, step_count)

    # Per-episode state (reset on every reset() call)
    self._turn                  # current turn number (1-indexed)
    self._phase                 # 1 or 2
    self._rewards               # list of all rewards this episode
    self._done                  # episode terminated?
    self._last_reward           # most recent reward value
    self._history               # rolling action history (last 3 turns)
    self._phase1_output         # degraded handoff passed to Phase 2 obs
    self._phase1_grading_output # original Phase 1 output used for grading
    self._turn_config           # current turn's attacker observation
    self._consecutive_wrong     # consecutive wrong Phase 2 actions
    self._last_obs              # cached last observation (for done/error)
    self._episode_replay        # full episode trace for metadata
    self._last_replay_strategy  # strategy from previous turn (shift detection)
    self._active_defenses       # list of active defense controls with TTL
    self._foothold_established  # bool: has attacker persisted on hard task?
    self._tool_trace            # full list of all tool calls this episode
    self._turn_tool_evidence    # verified tool evidence keyed by turn
    self._turn_tool_results     # all tool results keyed by turn
```

**Two phase1 outputs:** `_phase1_output` is the **degraded handoff** that the Phase 2 observation receives — on the hard task late turns, the analyst confidence gets artificially lowered to simulate real-world handoff degradation. `_phase1_grading_output` is the **original output** that the grader uses. This distinction ensures degraded handoffs don't unfairly penalize Phase 1 grading.

---

### 6.3 `reset()` — Episode Initialization

```python
def reset(self, task_name: str = None) -> AdaptShieldObservation:
```

1. Optionally switches task (hot-reload without destroying the env object)
2. Resets all episode state to initial values
3. Calls `self._attacker.reset_episode()` — resets stage index and turn counter
4. Calls `self._attacker.build_observation()` — builds turn 1 network state
5. Wraps with active defense alerts (empty on turn 1)
6. Calls `build_phase1_obs()` to construct the Phase 1 observation dict
7. Converts to `AdaptShieldObservation` via `_to_obs()`
8. Attaches metadata with available tools, active defenses, mission profile
9. Returns the observation

**Return invariant:** Always returns a Phase 1 observation. Episode always starts with the analyst turn.

---

### 6.4 `step()` — The Core Turn Loop

```python
def step(self, action: AdaptShieldAction | Phase1Action | Phase2Action) -> AdaptShieldObservation:
```

The step function handles two distinct paths based on `self._phase`:

#### Phase 1 → Phase 2 Transition

```python
if self._phase == 1:
    phase1_output = {
        "threat_type":        _action_value(action.threat_type, "unknown"),
        "confidence":         _action_float(action.confidence, 0.5),
        "target_node":        _action_value(action.target_node, "unknown"),
        "recommended_action": _action_value(action.recommended_action, "monitor"),
        "reasoning":          str(action.reasoning or ""),
    }
    self._phase1_grading_output = dict(phase1_output)          # original for grading
    self._phase1_output = _degrade_handoff(phase1_output, ...) # possibly degraded for Phase 2
    self._phase = 2
    # Build Phase 2 observation (blind to raw state)
    # Return with reward = last_reward (or 0.01 if no history)
```

No reward is computed at Phase 1 — the Phase 1 transition just collects the assessment and moves to Phase 2. The reward in the returned observation is the **previous turn's reward** (so the agent has feedback context).

#### Phase 2 → Grade → Advance Turn

```python
# Grade the full turn
reward, catastrophic, info = grade_step(
    phase1_action=self._phase1_grading_output,
    phase2_action=p2,
    turn_config=self._turn_config,
    stage=current_stage,
    consecutive_wrong=self._consecutive_wrong,
    task_name=self._task_name,
    foothold_established=self._foothold_established,
    mission_profile=self._mission_profile,
    tool_context=self._tool_context_for_turn(),
)
```

After grading:
1. `_register_active_defense(p2)` — if action has TTL, add to active defense registry
2. `_update_foothold_state()` — check if agent missed containment (hard task)
3. Update consecutive_wrong counter
4. Append reward to `self._rewards`
5. Append to `self._history` (rolling last-3)
6. Append to `self._episode_replay` (full episode trace)
7. Call `self._attacker.advance_turn(acted_correctly)` — possibly escalate stage
8. Call `self._decay_active_defenses()` — decrement TTLs
9. Increment `self._turn`, reset `self._phase = 1`
10. Check `episode_done = catastrophic or (turn > max_turns)`
11. Build next Phase 1 observation (or terminal observation)

---

### 6.5 Handoff Degradation

```python
def _degrade_handoff(phase1_output, turn_config, task_name, turn):
    output = dict(phase1_output)
    if (task_name == "polymorphic-zero-day" and
            turn >= 5 and
            turn_config.get("strategy") == "lateral_movement"):
        output["confidence"] = min(float(output.get("confidence", 0.5)), 0.42)
        output["recommended_action"] = "monitor"
        output["handoff_quality"] = "degraded"
        output["handoff_note"] = (
            "Analyst confidence degraded after attacker pivot; executor must decide "
            "whether monitor is too passive for lateral movement."
        )
    else:
        output["handoff_quality"] = "clean"
    return output
```

On the hard task, turns ≥ 5 with lateral movement cause the handoff to Phase 2 to be **artificially degraded**: confidence capped at 0.42 and `recommended_action` overridden to `"monitor"`. The grader still uses the original Phase 1 output. The Phase 2 executor must override a bad recommendation and choose `isolate` or `honeypot` despite the analyst suggesting passivity.

This tests **executor judgment under degraded analyst input** — a realistic scenario where the threat analyst has lower confidence and makes a conservative recommendation, but the executor should know better.

---

### 6.6 Active Defense System

Active defenses represent controls the agent has deployed. They persist across turns with a TTL (time-to-live):

```python
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
```

When an agent executes a defense action:

```python
def _register_active_defense(self, p2):
    action = p2.get("action", "monitor")
    if action not in DEFENSE_TTL:
        return  # monitor has no TTL
    self._active_defenses.append({
        "action": action,
        "target": target_node,
        "ttl": DEFENSE_TTL[action],
        "side_effect": DEFENSE_SIDE_EFFECT[action],
    })
```

Each turn, TTLs decay:
```python
def _decay_active_defenses(self):
    self._active_defenses = [
        {**control, "ttl": control["ttl"] - 1}
        for control in self._active_defenses
        if control["ttl"] - 1 > 0
    ]
```

Active defenses appear in the next turn's `active_alerts`:
```
[CONTROL] rate_limit active on auth_service (ttl=1, side_effect=login_latency)
```

This gives the agent feedback that its previous action is still in effect, preventing redundant re-application.

---

### 6.7 Foothold State (Hard Task Only)

```python
def _update_foothold_state(self, p2, info, stage):
    if (self._task_name != "polymorphic-zero-day" or
            self._foothold_established or
            stage not in ("exploit", "exfiltration")):
        return False  # not applicable

    if p2.get("action") == "monitor" or not info.get("acted_correctly", False):
        self._foothold_established = True
        return True  # foothold just established
    return False
```

Foothold triggers when:
- Task is `polymorphic-zero-day`
- Current stage is `exploit` or `exfiltration`
- Agent either monitors (too passive) or acts incorrectly

Once foothold is established, it persists for the rest of the episode and:
1. Changes the correct action for `lateral_movement` from `isolate` → `honeypot`
2. Injects a persistent `[FOOTHOLD]` alert into subsequent observations
3. Makes the EDR tool show beaconing and persistence indicators

---

### 6.8 Episode Replay

After every Phase 2 step, the environment records:

```python
self._episode_replay.append({
    "turn":                self._turn,
    "p1":                  threat_type_classified,
    "p2_action":           action_taken,
    "target":              target_node,
    "result":              "optimal" | "heavy" | "wrong" | "false_positive" | "unverified",
    "shift":               bool,           # strategy changed this turn?
    "impact":              float,          # business_impact score
    "blast_radius":        list[str],      # downstream affected services
    "active_defenses":     list,           # snapshot of active controls
    "foothold_established": bool,
    "foothold_transition": bool,           # foothold just established this turn?
    "mission_alignment":   str,
    "tool_calls":          int,
    "tool_evidence_found": bool,
})
```

This replay is attached to the terminal observation's metadata as `"episode_replay"`, enabling the `baseline.py --replay` flag to print a human-readable full episode trace.

---

## PART 7 — SOC Tool Layer

### 7.1 Tool Registration

Four SOC tools are registered in `AVAILABLE_SOC_TOOLS`:

```python
AVAILABLE_SOC_TOOLS = [
    {"name": "log_search",  "endpoint": "/tools/log_search",
     "description": "Search recent SIEM/application logs for a node and time window."},
    {"name": "cmdb_lookup", "endpoint": "/tools/cmdb_lookup",
     "description": "Inspect service ownership, criticality, dependencies, and blast radius."},
    {"name": "edr_status",  "endpoint": "/tools/edr_status",
     "description": "Check endpoint containment, persistence, beaconing, and active controls."},
    {"name": "vuln_lookup", "endpoint": "/tools/vuln_lookup",
     "description": "Query internal package/advisory risk for supply-chain investigations."},
]
```

These are exposed in `metadata["available_tools"]` on every observation, so the agent always knows what tools exist.

---

### 7.2 `call_tool()` — Public Interface

```python
def call_tool(self, tool_name: str, **params: Any) -> Dict[str, Any]:
```

The public method dispatches to internal tool methods, records the result, and returns a **sanitized public result** (no grader labels):

```python
def _public_tool_result(result):
    hidden = {"evidence_type", "verified"}
    public = {key: value for key, value in result.items() if key not in hidden}
    public["result_summary"] = _tool_summary(result)
    return public
```

The internal result has `evidence_type` and `verified` flags used by the grader. The public result strips these so the agent must **infer** from observable facts, not from ground truth labels.

---

### 7.3 `log_search` Tool

The most important tool. Returns SIEM-style event logs for a given node.

**How it works:**
- Compares the queried `node` to `correct_target` from the current turn config
- If `node == correct_target` AND strategy matches → returns attack-specific events with high confidence
- Otherwise → returns benign routine events

```python
# Brute force on auth_service:
events = [
    "auth_service: 143 failed logins grouped by /24 source subnet",
    "auth_service: password spray pattern across active user accounts",
    "payment_service: no matching checkout errors during same window",
]
confidence = 0.88

# Benign query (wrong node):
events = ["nodeX: routine health heartbeat accepted"]
confidence = 0.32
```

**Hard task difference:** Event descriptions are slightly more ambiguous for polymorphic-zero-day to make signal fusion harder:
```python
# Hard task lateral movement:
"auth_service: low-volume identity failures persist from earlier abuse window"
# vs easy task:
"payment_service: service account login from previously unseen host"
```

---

### 7.4 `cmdb_lookup` Tool

Returns service metadata: ownership, criticality, dependencies, safe actions.

```python
dependencies = {
    "auth_service":    ["payment_service"],
    "payment_service": ["api_gateway"],
    "database":        ["payment_service", "api_gateway"],
    "api_gateway":     ["auth_service", "payment_service", "database"],
}.get(node, [])

criticality = {
    "auth_service": "high", "payment_service": "critical",
    "database": "critical", "api_gateway": "high",
}.get(node, "unknown")

safe_actions = {
    "auth_service":    ["rate_limit", "patch", "monitor"],
    "payment_service": ["honeypot", "rate_limit", "monitor"],
    "database":        ["honeypot", "monitor"],
    "api_gateway":     ["patch", "rate_limit", "monitor"],
}.get(node, ["monitor"])
```

Notable: `database` is marked `critical` with no `isolate` in safe_actions — you don't take the database offline. `payment_service` can take `honeypot` but not `isolate` in the safe list (though `isolate` is still technically valid, just heavy-handed).

This tool is always `verified: True` regardless of node — it returns operational context, not attack evidence.

---

### 7.5 `edr_status` Tool

Returns endpoint containment and process behavior.

**Lateral movement on payment_service:**
```python
status = {
    "containment": "partial",         # if foothold established
    "persistence": True,              # if foothold established
    "beaconing":   True,
    "process_note": "unknown child process under service account context",
}
confidence = 0.87
```

**Hard task nuance:** Before foothold is established, EDR shows `"unconfirmed"` containment and `"unexpected child process... no confirmed beacon yet"` — more ambiguous, lower confidence (0.74). After foothold, it confirms persistence (0.87).

This makes the hard task EDR tool stateful — the same tool call on the same node returns different results depending on whether the attacker has established persistence.

**Active defenses integration:**
```python
active_controls = [
    control for control in self._active_defenses
    if control.get("target") == node
]
if active_controls:
    status["containment"] = "control_active"
    confidence = 0.70
```

If you've already deployed `rate_limit` on a node, EDR shows it as `"control_active"` — feedback that your previous defense is still running.

---

### 7.6 `vuln_lookup` Tool

Specialized for supply chain attacks. Only returns attack evidence when querying the correct node (`api_gateway`) for the correct strategy (`supply_chain`).

```python
if relevant:
    advisory = {
        "package":                  package or "gateway-router",
        "advisory_id":              "ADV-AS-042",
        "risk":                     "critical",
        "finding":                  "registry hash mismatch with unsigned update source",
        "recommended_mitigation":   "patch from trusted registry",
    }
    confidence = 0.91

else:
    advisory = {
        "package":                  package or "unknown",
        "advisory_id":              None,
        "risk":                     "none_known",
        "finding":                  "no matching active internal advisory",
        "recommended_mitigation":   "continue investigation",
    }
    confidence = 0.55
```

The `advisory_id: "ADV-AS-042"` is a fictional internal advisory. The agent must see `risk: "critical"` and `finding: "hash mismatch"` and infer this is supply chain. The grader checks that the agent called `vuln_lookup` AND `log_search` (the required fusion pair for supply_chain) before giving full credit.

---

### 7.7 Tool Evidence Recording

Every tool call is recorded internally in two structures:

```python
# All tool calls this episode (for metadata/trace)
self._tool_trace.append({
    "turn": turn, "phase": phase, "tool": tool_name,
    "node": node, "confidence": confidence,
    "summary": _tool_summary(result),
})

# Verified evidence keyed by turn (for grader)
if internal["verified"]:
    self._turn_tool_evidence.setdefault(turn, []).append(internal)
```

The `_tool_context_for_turn()` method assembles the grader context:
```python
{
    "turn": current_turn,
    "tool_count": number_of_tool_calls_this_turn,
    "evidence": verified_evidence_for_this_turn,
    "tool_results": all_results_for_this_turn,
}
```

This context is passed to `grade_step()`, which checks whether the required tool fusion was achieved.

---

## PART 8 — FastAPI Server (`app.py`)

### 8.1 Architecture

`app.py` is intentionally minimal — it's a thin HTTP wrapper over the environment class. The core env logic lives entirely in `adaptshield_environment.py`.

```python
app = create_app(
    make_env,
    AdaptShieldAction,
    AdaptShieldObservation,
    env_name="adaptshield",
    max_concurrent_envs=10,
)
```

`create_app` is from `openenv.core.env_server.http_server` — it wires up the standard OpenEnv endpoints (`/reset`, `/step`, `/state`) with session management, concurrency control, and proper error handling.

---

### 8.2 Factory Pattern (Critical Design Decision)

```python
def make_env() -> AdaptShieldEnvironment:
    """
    Factory function — fresh isolated instance per session.
    Never a singleton. Evaluator sessions must be independent.
    """
    return AdaptShieldEnvironment(task_name=DEFAULT_TASK)
```

The file header explicitly documents this as a Round 1 failure fix:

> **CRITICAL: Uses factory pattern (make_env function), NOT singleton. Singleton was the Round 1 failure — always served wrong task. Factory creates a fresh isolated instance per evaluator session.**

A singleton would share state between concurrent evaluator sessions — session A's episode state would leak into session B's responses. The factory pattern guarantees isolation.

---

### 8.3 Standard OpenEnv Endpoints

These are auto-generated by `create_app`:

| Endpoint | Method | Purpose |
|---|---|---|
| `/reset` | POST | Start new episode, returns Phase 1 observation |
| `/step` | POST | Submit action, returns next observation |
| `/state` | GET | Returns current `State` (episode_id, step_count) |
| `/health` | GET | Health check (used by Docker HEALTHCHECK) |

---

### 8.4 SOC Tool Endpoints

AdaptShield adds a persistent **SOC session** layer on top of the standard OpenEnv interface:

```python
@app.post("/soc/reset")
async def soc_reset(payload) -> Dict:
    task = payload.get("task", DEFAULT_TASK)
    env = AdaptShieldEnvironment(task_name=task)
    obs = env.reset()
    session_id = str(uuid4())
    SOC_SESSIONS[session_id] = env   # store env keyed by session_id
    return {"session_id": session_id, "observation": obs.model_dump(...)}

@app.post("/soc/step")
async def soc_step(payload) -> Dict:
    env = _soc_session(payload)      # lookup by session_id
    action = AdaptShieldAction(**payload["action"])
    obs = env.step(action)
    return {"observation": obs.model_dump(...), "reward": obs.reward, "done": obs.done}
```

The SOC sessions (`SOC_SESSIONS` dict) are separate from the OpenEnv sessions managed by `create_app`. This allows the inference script to use either path:

1. **OpenEnv path** (`/reset` + `/step`): Standard protocol, used by evaluators
2. **SOC path** (`/soc/reset` + `/soc/step` + `/tools/*`): Adds tool calls within the session

---

### 8.5 SOC Tool HTTP Endpoints

```python
@app.post("/tools/log_search")
async def tool_log_search(payload) -> Dict:
    return _soc_session(payload).call_tool(
        "log_search",
        node=payload.get("node"),
        query=payload.get("query", ""),
    )

@app.post("/tools/cmdb_lookup")
@app.post("/tools/edr_status")
@app.post("/tools/vuln_lookup")
```

All tool endpoints require a valid `session_id` in the payload. The session lookup:

```python
def _soc_session(payload):
    session_id = str(payload.get("session_id", ""))
    env = SOC_SESSIONS.get(session_id)
    if env is None:
        raise HTTPException(404, "Unknown SOC session. Call /soc/reset first.")
    return env
```

This means tool calls are **stateful** — they draw from the same env instance as the `soc/step` calls, so tool responses reflect the current attacker stage, active defenses, and foothold state.

---

### 8.6 Server Entry Point

```python
def main(host: str = "0.0.0.0", port: int = 7860) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=7860)
    main()
```

Port 7860 is the HuggingFace Spaces default. The `main()` function must exist as a named function (not just `if __name__` block) because the OpenEnv validator does a literal string check for `def main()` in the source file.

---

### 8.7 Docker Deployment

The Dockerfile uses a two-stage build:

```dockerfile
FROM ghcr.io/meta-pytorch/openenv-base:latest AS builder
WORKDIR /app/env

# Install uv if not present
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies using uv.lock
RUN uv sync --frozen --no-install-project --no-editable
RUN uv sync --frozen --no-editable

FROM ghcr.io/meta-pytorch/openenv-base:latest
COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

EXPOSE 7860
HEALTHCHECK CMD curl -f http://localhost:7860/health || exit 1
CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 7860"]
```

**Key design choices:**
- Two-stage build keeps the final image clean (builder tools not included)
- `uv.lock` ensures reproducible installs (`--frozen` flag)
- `PYTHONPATH` includes `/app/env` so `import models` and `import server.app` resolve correctly
- Health check on `/health` is required by OpenEnv evaluator infrastructure

---

### 8.8 OpenEnv YAML Declaration

`openenv.yaml` is the spec file that the `openenv validate` CLI reads:

```yaml
spec_version: 1
name: adaptshield
type: space
runtime: fastapi
app: server.app:app
port: 7860
tasks:
  - name: direct-triage
    difficulty: easy
    max_steps: 5
  - name: dual-pivot
    difficulty: medium
    max_steps: 6
  - name: polymorphic-zero-day
    difficulty: hard
    max_steps: 8
```

This is the machine-readable contract used by the evaluator to know which tasks exist, what difficulty they are, and how long episodes run.

---

*End of Part 4. Continue to Part 5: Training System, Inference, Client, Baseline & Testing.*
