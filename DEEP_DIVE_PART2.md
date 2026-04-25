# AdaptShield Deep Dive — Part 2: Data Models & Task System

---

## PART 2 — Pydantic Data Models (`models.py`)

### 2.1 Overview

`adaptshield/models.py` (244 lines) defines the complete typed contract between the agent and the environment. Every action the agent sends and every observation it receives is validated through these models. There are **no raw dicts** crossing the boundary — everything is Pydantic v2.

The file exports:
- `DefenseAction` — Enum of valid defensive actions
- `ThreatType` — Enum of known attack strategies
- `Phase1Action` — Threat Analyst output
- `Phase2Action` — Tactical Executor output
- `AdaptShieldAction` — Unified transport model (accepts either phase)
- `AdaptShieldObservation` — Full observation returned after each step

---

### 2.2 `DefenseAction` Enum

```python
class DefenseAction(str, Enum):
    RATE_LIMIT = "rate_limit"
    ISOLATE    = "isolate"
    HONEYPOT   = "honeypot"
    PATCH      = "patch"
    MONITOR    = "monitor"
```

**Why an Enum?** To prevent LLM hallucination from reaching the grader. If an LLM outputs `"block_traffic"` or `"quarantine"`, Pydantic rejects it at the boundary — the grader never sees garbage. Each value maps directly to a grader outcome:

| Action | Effect | Disruption Cost |
|---|---|---|
| `rate_limit` | Throttle traffic, keeps service online | 0.10 |
| `isolate` | Takes node completely offline | 0.35 |
| `honeypot` | Redirects attacker to decoy | 0.12 |
| `patch` | Applies security update | 0.06 |
| `monitor` | Observe only, no action | 0.00 |

---

### 2.3 `ThreatType` Enum

```python
class ThreatType(str, Enum):
    BRUTE_FORCE      = "brute_force"
    LATERAL_MOVEMENT = "lateral_movement"
    EXFILTRATION     = "exfiltration"
    SUPPLY_CHAIN     = "supply_chain"
    BENIGN           = "benign"
```

The Phase 1 agent classifies the current attacker strategy into one of these five categories. Each maps to a specific correct defensive action:

| Threat Type | Correct Action | Correct Target |
|---|---|---|
| `brute_force` | `rate_limit` | `auth_service` |
| `lateral_movement` | `isolate` | `payment_service` |
| `exfiltration` | `honeypot` | `database` |
| `supply_chain` | `patch` | `api_gateway` |
| `benign` | `monitor` | any |

---

### 2.4 `Phase1Action` — Threat Analyst Output

```python
class Phase1Action(Action):
    threat_type:        str           # identified attack strategy
    confidence:         float         # 0.0 to 1.0, validated by ge/le
    target_node:        str           # which node is under attack
    recommended_action: DefenseAction # what Phase 2 should do
    reasoning:          Optional[str] # chain-of-thought, not graded
```

**Key design decisions:**
- `reasoning` is optional and **never graded** — it exists purely to improve training stability. LLMs that reason before committing a JSON output are more stable.
- `confidence` has Pydantic `ge=0.0, le=1.0` constraints — a model outputting `confidence: 1.5` is rejected.
- `recommended_action` uses `DefenseAction` Enum — hallucinated action names cannot reach the environment.
- `threat_type` is a plain `str` (not enum) intentionally — the grader does string comparison, so subtle mismatches like `"brute force"` vs `"brute_force"` fail cleanly without an enum exception.

**What gets graded from Phase1Action:**
- `threat_type` == attacker's actual strategy → +0.15 reward
- `target_node` == correct target → +0.10 reward
- `recommended_action` — passed to Phase 2 as advice (not directly graded, but Phase 2's action is)

---

### 2.5 `Phase2Action` — Tactical Executor Output

```python
class Phase2Action(Action):
    action:      DefenseAction  # the defense to execute
    target_node: str            # which node to apply it to
    reasoning:   Optional[str]  # chain-of-thought, not graded
```

**Critical constraint:** Phase 2 agent **never sees** `network_nodes` or `active_alerts`. It only receives `phase1_assessment` (the Phase 1 output). This enforces information asymmetry — the executor must trust or interpret the analyst's handoff.

**What gets graded from Phase2Action:**
- `action` == optimal action for the threat → +0.39 reward
- `action` == heavy-handed but effective → +0.18 reward
- `action` == wrong → -0.25 reward
- Wrong target node with right action → partial credit (0.5 × heavy reward)

---

### 2.6 `AdaptShieldAction` — Unified Transport Model

This is the most architecturally interesting model. The OpenEnv protocol expects a **single** action model, but AdaptShield needs two different action shapes. The solution is a unified model with a `@model_validator`:

```python
class AdaptShieldAction(Action):
    # Phase 1 fields
    threat_type:        Optional[str]          = None
    confidence:         Optional[float]        = None
    recommended_action: Optional[DefenseAction]= None
    # Shared
    target_node:        Optional[str]          = None
    reasoning:          Optional[str]          = None
    # Phase 2 field
    action:             Optional[DefenseAction]= None
```

The `@model_validator(mode="after")` enforces mutual exclusivity:
1. If Phase 1 fields are present AND `action` is present → `ValueError` (cannot be both)
2. If neither Phase 1 nor Phase 2 fields → `ValueError` (must be one)
3. If Phase 1 shape → validates `threat_type`, `confidence`, `target_node`, `recommended_action` all present
4. If Phase 2 shape → validates `action` and `target_node` both present

**Why this design?** The alternative — two separate models — causes HTTP 500 errors when the OpenEnv evaluator sends a Phase 2 payload and the server tries to validate Phase 1 required fields. The unified model with explicit validator is the clean solution. (This is called out in the file's header comment as a "CRITICAL DESIGN DECISION".)

---

### 2.7 `AdaptShieldObservation` — What the Agent Receives

```python
class AdaptShieldObservation(Observation):
    scenario_id:        str                     # episode UUID
    task_name:          str                     # which task
    phase:              int                     # 1 or 2
    turn:               int                     # current turn number
    max_turns:          int                     # episode length
    network_nodes:      Dict[str, Any]          # EMPTY in Phase 2
    active_alerts:      List[str]               # EMPTY in Phase 2
    attack_stage:       str                     # recon/exploit/exfiltration/none
    history:            List[Dict[str, str]]    # last 3 turns
    phase1_assessment:  Optional[Dict[str, Any]]# Phase 2 only
    system_context:     str                     # system prompt for this phase
    available_actions:  List[str]               # always all 5 actions
    last_action_result: Optional[str]           # grader feedback string
    reward:             float                   # last step reward
    done:               bool                    # episode complete?
    metadata:           Dict[str, Any]          # normalized_score + breakdown
```

**Phase 1 observation:** `network_nodes` is populated (4 nodes with metrics). `active_alerts` has SIEM-style alert strings. `phase1_assessment` is `None`.

**Phase 2 observation:** `network_nodes = {}`. `active_alerts = []`. `phase1_assessment` contains the Phase 1 output dict. `attack_stage = "hidden"`.

**Critical invariant:** `metadata["normalized_score"]` is **always present** from step 1. The OpenEnv evaluator requires this, and every code path guarantees it — even error paths fall back to `0.50`.

**`model_dump` override:** The base `Observation` class excludes `metadata` from serialization. `AdaptShieldObservation` overrides `model_dump` to un-exclude it, because `normalized_score` lives in metadata and must reach the evaluator.

---

## PART 3 — Task System & Scenarios (`scenarios.py`)

### 3.1 Task Configurations

`scenarios.py` defines the three tasks in `TASK_CONFIGS`:

#### Task 1: `direct-triage` (Easy)
```python
{
    "max_turns": 5,
    "description": "Single fixed attacker strategy. Learn baseline threat response.",
    "mission_profile": {
        "mission_id": "login_stability",
        "primary_asset": "auth_service",
        "sla_priority": "availability",
        "risk_tolerance": "medium",
        "objective": "Preserve user login availability while stopping credential abuse.",
    }
}
```
- Attacker uses ONE strategy for the entire episode (no shifts)
- Rule baseline scores ~0.87
- Designed for initial learning — clean signal, predictable environment

#### Task 2: `dual-pivot` (Medium)
```python
{
    "max_turns": 6,
    "description": "Two strategies alternating every 20 episodes.",
    "mission_profile": {
        "mission_id": "checkout_continuity",
        "primary_asset": "payment_service",
        "sla_priority": "availability",
        "risk_tolerance": "medium",
        "objective": "Keep checkout online while containing internal movement.",
    }
}
```
- Attacker starts with `brute_force`, pivots to `lateral_movement` at turn 3
- Strategy also rotates between episodes (every 20 episodes)
- Distractor signals added — `auth_service` shows residual brute-force noise during lateral movement
- Rule baseline scores ~0.76

#### Task 3: `polymorphic-zero-day` (Hard)
```python
{
    "max_turns": 8,
    "description": "All four strategies with mid-episode shift and noise injection.",
    "mission_profile": {
        "mission_id": "breach_containment",
        "primary_asset": "database",
        "sla_priority": "containment",
        "risk_tolerance": "low",
        "objective": "Protect sensitive data and adapt if attacker persistence is established.",
    }
}
```
- All 4 strategies active
- Strategy shifts mid-episode after turn 3
- 15% chance of benign noise injection per turn (false positive traps)
- Foothold mechanic: if agent misses containment, attacker establishes persistence
- SOC tool evidence required for full credit (multi-tool fusion)
- Rule baseline scores ~0.52

---

### 3.2 Phase 1 System Prompt

The system prompt injected into Phase 1 observations:

```
You are a Threat Analyst defending a 4-node enterprise network.
Analyze the network metrics and security alerts to classify the threat.

Network nodes: auth_service, payment_service, database, api_gateway

Attack strategies to recognize:
- brute_force: High failed login rate on auth_service, error spikes
- lateral_movement: Unusual cross-node connections, service account anomalies
- exfiltration: Anomalous outbound data volume from database node
- supply_chain: Suspicious update requests or hash mismatches on api_gateway
- benign: Normal operations — scheduled jobs, routine activity

Respond ONLY with valid JSON, no other text:
{"threat_type": "...", "confidence": 0.0, "target_node": "...", "recommended_action": "...", "reasoning": "..."}
```

**Design choices:**
- No MITRE codes — forces semantic reasoning, not pattern matching
- No explicit strategy names in the alert text — agent must infer from metrics
- JSON-only response — parseable for grading without NLP

---

### 3.3 Phase 2 System Prompt

```
You are a Tactical Executor defending a 4-node enterprise network.
You receive a threat assessment from the Threat Analyst and must execute the defense.
You CANNOT see raw network logs — act only on the assessment provided.

Available actions:
- rate_limit: Throttle traffic. Light touch, keeps service online. Best for DoS/brute force.
- isolate: Take node offline. Stops spread but causes downtime. Use for lateral movement.
- honeypot: Redirect attacker to decoy. Best for data exfiltration attempts.
- patch: Apply security update. Targeted fix for supply chain attacks.
- monitor: Observe without acting. Use only when genuinely uncertain or event is benign.

Respond ONLY with valid JSON:
{"action": "...", "target_node": "...", "reasoning": "..."}
```

**Design choices:**
- Phase 2 agent is told it cannot see raw logs — enforces correct mental model
- Action descriptions are semantically correct — no misleading hints
- `monitor` is described as valid for uncertainty — avoids always-act bias

---

### 3.4 Mission Context Injection

Every system prompt gets mission context appended via `_with_mission_context()`:

```python
def _with_mission_context(system_prompt, mission_profile):
    mission = "\n".join([
        "Mission context:",
        f"- mission_id: {mission_profile['mission_id']}",
        f"- primary_asset: {mission_profile['primary_asset']}",
        f"- sla_priority: {mission_profile['sla_priority']}",
        f"- risk_tolerance: {mission_profile['risk_tolerance']}",
        f"- objective: {mission_profile['objective']}",
    ])
    return f"{system_prompt}{mission}"
```

This means the agent always knows:
- What service matters most
- Whether uptime or containment is the priority
- The operational objective in plain English

This enables the grader's `mission_alignment` scoring — actions that violate the SLA (e.g., isolating `auth_service` when SLA priority is availability) get penalized.

---

### 3.5 Observation Builders

Two builder functions construct observation dicts from raw environment state:

**`build_phase1_obs()`** — Full state visible:
- Populates `network_nodes`, `active_alerts`, `attack_stage`
- `history` is last 3 turns only (rolling window)
- `phase1_assessment` = `None`
- `metadata.normalized_score` = 0.50 (default until first step)

**`build_phase2_obs()`** — Deliberately blinded:
- `network_nodes = {}` (empty dict)
- `active_alerts = []` (empty list)
- `attack_stage = "hidden"` (explicit string)
- `phase1_assessment` = Phase 1 output dict (the handoff)
- `metadata.normalized_score` = current episode score

This asymmetry is the core mechanic — the information bottleneck between analyst and executor.

---

*End of Part 2. Continue to Part 3: The Attacker Engine & Grader.*
