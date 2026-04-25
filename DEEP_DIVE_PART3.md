# AdaptShield Deep Dive — Part 3: Attacker Engine & Grader

---

## PART 4 — The Attacker Engine (`attacker.py`)

### 4.1 Design Philosophy

The attacker is a **scripted Python deterministic adversary** — never an LLM. This is a deliberate and critical design decision. Using an LLM as an attacker would:

1. Make grading non-deterministic
2. Add enormous inference cost
3. Create training instability (reward signal depends on another model's behavior)
4. Make reproducibility impossible

Instead, the attacker is a finite state machine: it picks a strategy, builds network observations, and escalates attack stages based on whether the agent responds correctly.

---

### 4.2 Strategy Map

```python
STRATEGY_MAP = {
    "brute_force": {
        "correct_action": "rate_limit",
        "correct_target": "auth_service",
        "description": "Credential stuffing attack on authentication service",
    },
    "lateral_movement": {
        "correct_action": "isolate",
        "correct_target": "payment_service",
        "description": "Attacker pivoting from compromised host to internal nodes",
    },
    "exfiltration": {
        "correct_action": "honeypot",
        "correct_target": "database",
        "description": "Slow sustained data exfiltration from database",
    },
    "supply_chain": {
        "correct_action": "patch",
        "correct_target": "api_gateway",
        "description": "Malicious package injected into update pipeline",
    },
}
```

Each strategy has exactly one correct action and one correct target. This creates a clean mapping the agent must learn:

```
brute_force      → rate_limit    on auth_service
lateral_movement → isolate       on payment_service
exfiltration     → honeypot      on database
supply_chain     → patch         on api_gateway
```

---

### 4.3 Strategy Rotation

```python
STRATEGY_ROTATION = [
    "brute_force",
    "lateral_movement",
    "exfiltration",
    "supply_chain",
]
```

Between episodes, the strategy rotates on a **fixed schedule** per task:

| Task | `_shift_every` | Behavior |
|---|---|---|
| `direct-triage` | 9999 | Strategy never changes (same strategy every episode) |
| `dual-pivot` | 20 | Rotates after every 20 episodes |
| `polymorphic-zero-day` | 10 | Rotates after every 10 episodes |

**Mid-episode shift (polymorphic-zero-day only):**
```python
if self.task_name == "polymorphic-zero-day" and self._turn >= 3:
    base_idx = (base_idx + 1) % len(STRATEGY_ROTATION)
```
After turn 3 within any episode, the strategy index shifts by +1. This means:
- Turns 1-2: `brute_force`
- Turns 3+: `lateral_movement` (or next in rotation)

The agent must detect this shift from metrics alone — it never receives `"strategy changed"` explicitly.

**Dual-pivot mid-episode shift:**
```python
if self.task_name == "dual-pivot" and self._turn >= 3:
    base_idx = (base_idx + 1) % 2
```
Cycles only within the first 2 strategies (`brute_force` ↔ `lateral_movement`).

---

### 4.4 Attack Stage Progression

```python
ATTACK_STAGES = ["recon", "exploit", "exfiltration"]
```

Within every episode, the attack escalates through stages **only if the agent fails to act correctly**:

```python
def advance_turn(self, agent_acted_correctly: bool) -> None:
    self._turn += 1
    if not agent_acted_correctly:
        self._escalated = True
        self._stage_idx = min(self._stage_idx + 1, len(ATTACK_STAGES) - 1)
```

- If agent acts correctly → stage stays the same
- If agent acts wrong → `_stage_idx` increments (recon → exploit → exfiltration)
- Stage caps at `exfiltration` — cannot escalate past it

The stage multiplier amplifies all network signals:
```python
stage_multiplier = {"recon": 1.0, "exploit": 1.6, "exfiltration": 2.4}
```

So a brute force attack at `recon` shows 120 failed logins/min; at `exfiltration` stage it shows 288 failed logins/min. The signals get louder and harder to ignore.

---

### 4.5 Baseline Network State

```python
BASELINE_NODES = {
    "auth_service":    {"status": "healthy", "request_rate": 120, "error_rate": 0.01, "cpu": 22, "outbound_mb": 2},
    "payment_service": {"status": "healthy", "request_rate": 85,  "error_rate": 0.01, "cpu": 31, "outbound_mb": 5},
    "database":        {"status": "healthy", "request_rate": 34,  "error_rate": 0.01, "cpu": 18, "outbound_mb": 8},
    "api_gateway":     {"status": "healthy", "request_rate": 203, "error_rate": 0.02, "cpu": 29, "outbound_mb": 3},
}
```

This is the baseline "all clear" state. Attack signals are injected on top of this baseline as deltas. The agent must learn which deviations are suspicious.

---

### 4.6 Signal Generation Per Strategy

For each strategy, the attacker modifies specific node metrics and generates alert strings. Key patterns:

**Brute Force** (targets `auth_service`):
- `request_rate`: 340× multiplier
- `error_rate`: up to 48%
- `cpu`: up to 96%
- Alert: `"N failed login attempts from subnet X.X.X.x"`

**Lateral Movement** (targets `payment_service`):
- `payment_service.status` → `"suspicious"`
- `auth_service` elevated as distractor (on dual-pivot and hard task)
- Alert: `"service account 'X' authenticated from new host"`

**Exfiltration** (targets `database`):
- `outbound_mb`: 12× multiplier on top of stage multiplier
- `request_rate`: 3× above baseline
- Alert: `"sequential reads from table 'X'"`

**Supply Chain** (targets `api_gateway`):
- `api_gateway.status` → `"suspicious"`
- Alert: `"binary hash mismatch on dependency 'X'"`

**Benign** (15% chance on hard task only):
- No attack signals
- Alert: `"scheduled database backup completed"` or similar
- `api_gateway.cpu` elevated as decoy
- `correct_action = "monitor"`, `correct_target = "none"`

---

### 4.7 Alert Format

```python
def _alert_prefix(self) -> str:
    source = random.choice(["SIEM", "EDR", "WAF", "NETFLOW"])
    alert_id = random.randint(1000, 9999)
    minute = random.randint(0, 59)
    second = random.randint(0, 59)
    return f"[{source}-{alert_id} 03:{minute:02d}:{second:02d}Z]"
```

Alerts look like real SOC output: `[SIEM-4237 03:14:22Z] auth_service: 192 failed logins...`

The `ADAPTSHIELD_SEED` environment variable seeds the RNG, making the entire episode sequence fully reproducible for testing and baselines.

---

### 4.8 Noise Rate

```python
self._noise_rate = 0.15 if task_name == "polymorphic-zero-day" else 0.0
```

Only the hard task has noise. 15% of turns generate a **convincing but benign** observation — a false positive trap. If the agent acts aggressively on a benign event, it receives `FALSE_POSITIVE = -0.39` penalty. This teaches the agent to be discriminating rather than always acting.

---

## PART 5 — The Grader (`grader.py`)

### 5.1 Design Philosophy

The grader is **fully deterministic** — no LLMs, no NLP, no external API calls. It runs in milliseconds. Every reward is computed by pure Python string comparison and arithmetic.

The invariant: `all rewards ∈ [0.01, 0.99]` — the `_clamp()` function enforces this on every return path.

```python
def _clamp(value: float) -> float:
    return max(0.01, min(0.99, round(value, 2)))
```

---

### 5.2 Reward Constants

```python
BASE_REWARD         =  0.50   # agent gets this just for staying alive
P1_TYPE_BONUS       =  0.15   # correct threat type classification
P1_TARGET_BONUS     =  0.10   # correct target node identification
P2_OPTIMAL          =  0.39   # correct + efficient action
P2_HEAVY            =  0.18   # correct but over-aggressive action
P2_UNVERIFIED       =  0.10   # right action but no SOC tool evidence (hard task)
P2_WRONG            = -0.25   # wrong action on real threat
FALSE_POSITIVE      = -0.39   # acted aggressively on benign event
STAGE_ESCALATION    = -0.10   # attack stage escalated due to missed response
CATASTROPHIC        = -0.49   # database exfiltration completed
```

**Maximum possible single-step reward:**
`BASE_REWARD + P1_TYPE_BONUS + P1_TARGET_BONUS + P2_OPTIMAL = 0.50 + 0.15 + 0.10 + 0.39 = 1.14` → clamped to `0.99`

**Minimum possible single-step reward:**
`BASE_REWARD + CATASTROPHIC = 0.50 - 0.49 = 0.01` → plus operational and mission penalties

---

### 5.3 Reward Computation Flow

The `grade_step()` function processes in this order:

```
1. Is the event benign?
   YES → check if agent acted (false positive) or monitored (correct)
   NO  → continue to threat grading

2. Phase 1 accuracy bonuses
   - threat_type correct? → +0.15
   - target_node correct? → +0.10

3. Catastrophic failure check
   - strategy == exfiltration AND stage == exfiltration AND wrong action?
   → reward = 0.01, done = True

4. Stage escalation penalty
   - exploit stage + 1+ consecutive wrong → -0.10
   - exfiltration stage + 2+ consecutive wrong → -0.20

5. Phase 2 action grading
   - optimal action + correct target → +0.39
   - optimal action + wrong target  → +0.09 (half of heavy)
   - heavy action + correct target  → +0.18
   - anything else                  → -0.25

6. Hard task: tool verification check
   - correct action but no SOC evidence → +0.10 (not +0.39)

7. Operational impact penalty (capped at 0.05)

8. Mission alignment adjustment (capped at ±0.04)

9. _clamp() → final reward in [0.01, 0.99]
```

---

### 5.4 Optimal vs. Heavy-Handed Actions

```python
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
```

`isolate` is always the "blunt instrument" — it works in all cases but causes service downtime. Using it when a lighter action would suffice gets only +0.18 instead of +0.39.

---

### 5.5 Catastrophic Failure Logic

```python
EXFIL_CATASTROPHIC_ACTIONS = ["monitor", "rate_limit", "patch"]

if (strategy == "exfiltration" and
        stage == "exfiltration" and
        p2_action in EXFIL_CATASTROPHIC_ACTIONS):
    reward = BASE_REWARD + CATASTROPHIC  # = 0.01
    catastrophic = True
    done = True
```

If an exfiltration attack reaches the final stage AND the agent uses any action other than `honeypot` or `isolate`, the episode terminates immediately. This represents the attacker successfully completing data theft.

The agent must learn: `exfiltration` strategy + late stage = act decisively with `honeypot`.

---

### 5.6 Operational Impact Model

```python
ASSET_CRITICALITY = {
    "auth_service":    0.70,
    "payment_service": 0.90,
    "database":        1.00,
    "api_gateway":     0.80,
}

SERVICE_DEPENDENCIES = {
    "auth_service":    ["payment_service"],
    "payment_service": ["api_gateway"],
    "database":        ["payment_service", "api_gateway"],
    "api_gateway":     ["auth_service", "payment_service", "database"],
}

ACTION_DISRUPTION = {
    "monitor": 0.00, "patch": 0.06, "rate_limit": 0.10,
    "honeypot": 0.12, "isolate": 0.35,
}
```

The business impact is computed as:

```python
availability = min(1.0, disruption * (criticality + dependency_factor))
security     = _security_risk(result_kind, strategy, stage)
impact       = min(1.0, availability + security)
penalty      = min(0.05, impact * 0.05)  # capped at 0.05
```

`dependency_factor = min(1.0, 0.15 × number_of_dependents)` — nodes with more downstream dependents get a higher multiplier.

The penalty is intentionally **capped at 0.05 per turn** — enough to influence reward signal without destabilizing training curves.

---

### 5.7 Mission Alignment Scoring

```python
def _apply_mission_alignment(info, action, target, result_kind, mission_profile):
    sla_priority   = mission_profile.get("sla_priority", "balanced")
    primary_asset  = mission_profile.get("primary_asset", "unknown")
    risk_tolerance = mission_profile.get("risk_tolerance", "medium")

    if sla_priority == "availability" and action == "isolate" and target == primary_asset:
        adjustment -= 0.04   # SLA violation: isolating the key service
    elif sla_priority == "availability" and result_kind == "optimal" and action in ("rate_limit", "patch", "monitor"):
        adjustment += 0.02   # SLA aligned: light touch on availability mission
    elif sla_priority == "containment" and result_kind == "optimal" and action in ("honeypot", "isolate", "patch"):
        adjustment += 0.02   # containment aligned: decisive action on breach mission
    elif risk_tolerance == "low" and result_kind in ("wrong", "wrong_target"):
        adjustment -= 0.02   # risk misaligned: wrong action on low-tolerance mission
```

**Practical implications:**
- On `direct-triage` (auth_service, availability): Isolating `auth_service` costs -0.04 even if it stops the attack
- On `polymorphic-zero-day` (database, containment): Using `honeypot` or `isolate` correctly gives +0.02 bonus
- Low risk-tolerance missions are unforgiving of wrong actions

---

### 5.8 Tool Verification (Hard Task Only)

```python
REQUIRED_TOOL_FUSION = {
    "brute_force":      {"log_search", "cmdb_lookup"},
    "lateral_movement": {"edr_status", "log_search"},
    "exfiltration":     {"log_search", "edr_status"},
    "supply_chain":     {"vuln_lookup", "log_search"},
}
```

On `polymorphic-zero-day`, if the agent takes the **correct action** without calling the **required tool pair** and getting verified evidence, it receives only `P2_UNVERIFIED = +0.10` instead of `P2_OPTIMAL = +0.39`.

The grader checks:
1. Did the agent call at least the required tools for this strategy?
2. Did the tool calls return verified evidence pointing to the correct target?

Both must be true for full credit. This forces the agent to **investigate before acting** on the hard task.

---

### 5.9 Episode Score Normalization

```python
def normalize_episode_score(rewards: List[float]) -> float:
    if not rewards:
        return 0.50
    total    = sum(rewards)
    n        = len(rewards)
    max_poss = n * (BASE_REWARD + P2_OPTIMAL + P1_TYPE_BONUS + P1_TARGET_BONUS)  # 1.14 per step
    min_poss = n * (BASE_REWARD + CATASTROPHIC)  # 0.01 per step
    raw = (total - min_poss) / (max_poss - min_poss)
    return _clamp(raw)
```

The episode score is normalized relative to the theoretical best and worst possible outcomes for that episode length, then clamped to `[0.01, 0.99]`. This makes scores comparable across tasks with different `max_turns`.

**Baseline scores at ADAPTSHIELD_SEED=42:**

| Task | Score | Status |
|---|---|---|
| `direct-triage` | 0.870 | PASS |
| `dual-pivot` | 0.760 | PASS |
| `polymorphic-zero-day` | 0.520 | PASS |

The difficulty staircase (`0.87 → 0.76 → 0.52`) validates that the tasks get progressively harder as designed.

---

### 5.10 Contextual Countermeasure (Polymorphic Task)

When the attacker establishes a **foothold** (agent missed containment during lateral movement), the optimal action changes:

```python
if (task_name == "polymorphic-zero-day" and
        foothold_established and
        strategy == "lateral_movement"):
    correct_action = "honeypot"  # was "isolate"
```

This is "context-aware" reward — once the attacker has a persistent foothold, isolating the node is too aggressive (you'd cut off your own visibility). The correct move becomes `honeypot` (redirect and track). The grader emits `"Context-aware optimal"` in the score reason when this path activates.

---

*End of Part 3. Continue to Part 4: Environment Class, SOC Tools, and Server.*
