# AdaptShield Deep Dive — Part 5: Training, Inference, Client & Testing

---

## PART 9 — Training System (`train.py`, `train_smoke.py`)

### 9.1 Overview

AdaptShield has **three training paths**, each serving a different purpose:

| Path | File | GPU Required | Purpose |
|---|---|---|---|
| Smoke training | `train_smoke.py` | No | Verify env runs learning loops, zero deps |
| Policy Gradient | `train.py --trainer pg` | Yes (unsloth) | Safe fallback, simple REINFORCE |
| GRPO | `train.py --trainer grpo` | Yes (unsloth + TRL) | Full RL training with group relative optimization |

The `--trainer auto` flag prefers GRPO and falls back to PG if TRL is not installed.

---

### 9.2 Model Support

```python
MODEL_CHOICES = {
    "0.5b": "unsloth/Qwen2.5-0.5B-Instruct",
    "1.5b": "unsloth/Qwen2.5-1.5B-Instruct",
    "3b":   "unsloth/Qwen2.5-3B-Instruct",
    "7b":   "unsloth/Qwen2.5-7B-Instruct",
}
DEFAULT_MODEL = "1.5b"
MAX_SEQ_LEN   = 2048
LORA_RANK     = 16
```

All models are Qwen 2.5 Instruct variants loaded via unsloth's `FastLanguageModel` with 4-bit quantization. LoRA is applied to all projection matrices:

```python
model = FastLanguageModel.get_peft_model(
    model,
    r=LORA_RANK,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=LORA_RANK * 2,   # 32
    lora_dropout=0.0,
    bias="none",
    use_gradient_checkpointing="unsloth",
)
```

4-bit quantization + unsloth's gradient checkpointing allows training 7B models on single consumer GPUs.

---

### 9.3 Curriculum Learning

```python
CURRICULUM_STAGES = [
    ("direct-triage",        0.30),  # first 30% of episodes
    ("dual-pivot",           0.40),  # next 40%
    ("polymorphic-zero-day", 0.30),  # final 30%
]
```

The `task_for_episode()` function determines which task runs for each episode:

```python
def task_for_episode(episode, total_episodes, selected_task, curriculum):
    if not curriculum:
        if selected_task == "all":
            return TASKS[(episode - 1) % len(TASKS)], "round_robin"
        return selected_task, "fixed"

    progress = episode / total_episodes
    cumulative = 0.0
    for task, fraction in CURRICULUM_STAGES:
        cumulative += fraction
        if progress <= cumulative:
            return task, f"curriculum:{task}"
```

**Curriculum rationale:**
1. Start on the easy task (30%) — the agent learns the basic action-reward mapping
2. Move to medium (40%) — learns to handle strategy shifts
3. Finish on hard (30%) — learns multi-tool evidence fusion and adaptation

Without curriculum, throwing the agent at `polymorphic-zero-day` from episode 1 produces noisy rewards with slow learning. The curriculum provides a shaped learning signal.

---

### 9.4 Prompt Construction

Two system prompts are defined for training (slightly different from inference — training prompts emphasize tool evidence):

```python
PHASE1_SYS = """You are a Threat Analyst for a 4-node enterprise network.
...
If SOC tool evidence is provided, use it to update your belief before classifying.
Respond ONLY with valid JSON:
{"threat_type":"...","confidence":0.0,"target_node":"...","recommended_action":"...","reasoning":"..."}"""

PHASE2_SYS = """You are a Tactical Executor. Act only on the analyst handoff.
You cannot see raw network data in Phase 2.
Use the analyst handoff plus any SOC tool trace from this turn.
Respond ONLY with valid JSON:
{"action":"...","target_node":"...","reasoning":"..."}"""
```

**`make_phase1_prompt(obs)`** builds the user turn:
```
Network nodes:
{"auth_service": {...}, ...}

Active alerts:
[SIEM-4237 03:14:22Z] auth_service: 192 failed logins...

SOC tool evidence:
{"tool":"log_search","node":"auth_service","events":["..."]}

Recent history:
[{"turn": "1", "p1": "classified:brute_force", ...}]

Classify the threat:
```

**`make_phase2_prompt(obs)`** builds the Phase 2 user turn:
```
Threat assessment from analyst:
{"threat_type": "brute_force", "confidence": 0.9, ...}

SOC tool trace for this turn:
[{"tool": "log_search", "node": "auth_service", "summary": "..."}]

Choose the defensive action:
```

---

### 9.5 Policy Gradient Training Loop

```python
def train_policy_gradient(args):
    for episode in range(1, args.episodes + 1):
        task, stage = task_for_episode(episode, ...)
        samples, metrics = run_model_episode(model, tokenizer, task, ...)
        rewards = [sample["reward"] for sample in samples]
        baseline = sum(rewards) / len(rewards)  # mean baseline

        for sample in samples:
            advantage = sample["reward"] - baseline
            full_text = sample["prompt"] + sample["response"] + tokenizer.eos_token
            inputs = tokenizer(full_text, return_tensors="pt", ...)
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss * (-advantage)   # REINFORCE: maximize advantage
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
```

This is vanilla **REINFORCE** (policy gradient):
- `advantage = reward - baseline` — how much better than average was this action?
- `loss = -log_prob * advantage` — maximize probability of high-advantage actions
- Gradient clipping at 1.0 prevents exploding gradients
- Mean reward baseline reduces variance

**Checkpointing:**
```python
if row["score"] > best_score:
    best_score = row["score"]
    model.save_pretrained(output_dir / "best")
    tokenizer.save_pretrained(output_dir / "best")

if args.save_every and episode % args.save_every == 0:
    model.save_pretrained(output_dir / f"checkpoint-{episode}")
```

---

### 9.6 GRPO Training

GRPO (Group Relative Policy Optimization) is the more sophisticated training path. It requires building a **prompt bank** first, then training with TRL's `GRPOTrainer`.

#### Prompt Bank Construction

```python
def build_prompt_bank(tokenizer, selected_task, curriculum, rollout_episodes, max_steps, use_tools, seed):
    rows = []
    for episode in range(1, rollout_episodes + 1):
        env = AdaptShieldEnvironment(task_name=task)
        obs = env.reset()
        while not obs.done:
            # Run tool investigation
            tool_results = investigate_local_with_depth(env, obs, use_tools, thorough=True)
            obs_dict = attach_tool_results(obs_to_dict(obs), tool_results)
            messages = build_messages(obs_dict)
            # Get reference (ground truth) action
            reference = _current_reference(env)
            rows.append({
                "prompt":                     render_messages(messages, tokenizer),
                "task":                       task,
                "phase":                      phase,
                "turn":                       turn,
                "expected_threat_type":       reference["threat_type"],
                "expected_target_node":       reference["target_node"],
                "expected_recommended_action": reference["expected_action"] if phase == 1 else "",
                "expected_action":            reference["expected_action"] if phase == 2 else "",
                "tool_calls":                 len(tool_results),
            })
            # Step with the TEACHER policy (ground truth)
            obs = env.step(AdaptShieldAction(**_teacher_payload(phase, reference)))
```

The prompt bank is built using a **teacher policy** (reference policy that always takes the correct action). This generates high-quality training prompts covering all tasks, phases, and attack stages.

#### GRPO Reward Function

```python
def build_grpo_reward_fn():
    def reward_fn(completions, **kwargs) -> List[float]:
        for completion in completions:
            phase = kwargs["phase"][i]
            parsed = parse_response(text, phase)
            if phase == 1:
                reward = _phase1_reward(parsed, expected_threat_type, expected_target_node, expected_recommended_action)
            else:
                reward = _phase2_reward(parsed, expected_action, expected_target_node, tool_calls)
        return rewards
    return reward_fn
```

**Phase 1 reward breakdown:**
```python
def _phase1_reward(parsed, expected_threat_type, expected_target_node, expected_recommended_action):
    reward = 0.08                   # base
    if threat_type correct: += 0.36
    if target_node correct: += 0.20
    if recommended_action correct: += 0.18
    if confidence in [0, 1]: += 0.05
    if threat_type correct and confidence >= 0.65: += 0.06  # calibration bonus
    if threat_type wrong and confidence >= 0.80: -= 0.05    # overconfidence penalty
    if recommended_action == "monitor" and not benign: -= 0.05  # passivity penalty
```

**Phase 2 reward breakdown:**
```python
def _phase2_reward(parsed, expected_action, expected_target_node, tool_calls):
    reward = 0.08                   # base
    if action correct: += 0.62
    if target_node correct: += 0.18
    if action correct and tool_calls >= 2: += 0.07   # evidence fusion bonus
    if action == "monitor" and expected != "monitor": -= 0.08
```

Note: These GRPO rewards are **training-time rewards** computed from ground-truth labels. The **online environment reward** (from `grade_step()`) is the evaluation-time reward. They are aligned but not identical.

---

### 9.7 Smoke Training (`train_smoke.py`)

The smoke training is **dependency-free** — no GPU, no PyTorch, no unsloth. It uses a tiny tabular Q-learning policy to verify the environment runs learning loops correctly.

```python
class TabularDefensePolicy:
    def __init__(self, epsilon, lr):
        self.q = {
            threat: {action: 0.50 for action in ACTION_SPACE}
            for threat in POLICY
        }

    def choose_phase2(self, obs):
        threat = obs.phase1_assessment["threat_type"]
        if random.random() < self.epsilon:
            target, action = random.choice(ACTION_SPACE)   # explore
        else:
            target, action = argmax(self.q[threat])        # exploit

    def update(self, threat, selected, reward):
        old = self.q[threat][selected]
        self.q[threat][selected] = old + self.lr * (reward - old)  # Q-update

    def decay(self, rate, floor):
        self.epsilon = max(floor, self.epsilon * rate)
```

Default params: `epsilon=0.85`, `epsilon_decay=0.94`, `epsilon_floor=0.08`, `lr=0.35`.

The smoke training verifies:
1. The environment loop runs for N episodes without crashing
2. `normalized_score` is present in every observation metadata
3. The score improves over time (basic sanity check that rewards are shaping)
4. Output CSV can be loaded by `plot_training.py`

---

### 9.8 Training Outputs

GPU training writes to `checkpoints/<run-name>/`:

```
best/              ← saved at episode with highest episode score
  adapter_model.safetensors
  tokenizer_config.json
  ...
final/             ← saved at episode N (last episode)
checkpoint-{N}/    ← periodic checkpoints (if --save-every set)
metrics.json       ← full training log
```

**`metrics.json` structure:**
```json
{
  "model": "unsloth/Qwen2.5-1.5B-Instruct",
  "episodes": 60,
  "curriculum": true,
  "curriculum_stages": [["direct-triage", 0.3], ...],
  "use_tools": true,
  "trainer": "grpo",
  "prompt_bank_size": 120,
  "rows": [
    {"episode": 1, "task": "direct-triage", "stage": "curriculum:direct-triage",
     "score": 0.71, "steps": 10, "tool_calls": 3, "loss": 0.42, "seconds": 18.3},
    ...
  ],
  "evaluation_rows": [
    {"episode": 1, "task": "direct-triage", "stage": "evaluation", "score": 0.88, ...},
    {"episode": 2, "task": "dual-pivot",    "stage": "evaluation", "score": 0.79, ...},
    {"episode": 3, "task": "polymorphic-zero-day", "stage": "evaluation", "score": 0.63, ...},
  ],
  "best_score": 0.91
}
```

---

## PART 10 — Inference System (`inference.py`)

### 10.1 Overview

`inference.py` is the **evaluator-facing** script. It runs a complete episode using an external LLM via the OpenAI SDK (pointing to a LiteLLM proxy or HuggingFace router). It emits mandatory stdout format for the evaluator to parse.

All credentials come from environment variables — never hardcoded:

```python
API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME    = os.environ.get("ADAPTSHIELD_TASK", "direct-triage")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL", "http://localhost:7860")
```

---

### 10.2 Mandatory Stdout Format

```python
def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done} error={error}")

def log_end(success, steps, score, rewards):
    rs = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success} steps={steps} score={score:.3f} rewards={rs}")
```

Example output:
```
[START] task=direct-triage env=adaptshield model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action={"threat_type":"brute_force","confidence":0.9,...} reward=0.64 done=false error=null
[STEP] step=2 action={"action":"rate_limit","target_node":"auth_service"} reward=0.89 done=false error=null
...
[END] success=true steps=10 score=0.873 rewards=0.64,0.89,...
```

The evaluator parses `[START]`, `[STEP]`, and `[END]` lines to extract metrics.

---

### 10.3 Two Episode Paths

The inference script supports two execution paths:

**OpenEnv Path** (standard, no tools):
```python
def run_openenv_episode(client):
    env = AdaptshieldEnv(base_url=ENV_BASE_URL).sync()
    with env:
        result = env.reset(task_name=TASK_NAME)
        for step in range(MAX_STEPS):
            parsed = get_action(client, obs)
            action = build_env_action(parsed, phase)
            sr = env.step(action)
            obs = sr.observation
```

Uses the `AdaptshieldEnv` WebSocket client for persistent connection. Good for `direct-triage` and `dual-pivot`.

**SOC Path** (tool-aware, for hard task):
```python
def run_soc_episode(client, use_tools):
    reset = env_post("/soc/reset", {"task": TASK_NAME})
    session_id = reset["session_id"]
    for step in range(MAX_STEPS):
        tool_results = investigate_http(ENV_BASE_URL, session_id, obs, use_tools)
        obs_for_model = attach_tool_results(obs, tool_results)
        parsed = get_action(client, obs_for_model)
        result = env_post("/soc/step", {"session_id": session_id, "action": action_payload})
```

Uses HTTP POST calls (no WebSocket). Runs SOC tool investigation before each Phase 1 step. The `investigate_http()` function in `soc_tools.py` decides which tools to call based on network metrics.

**Path selection:**
```python
def should_use_tools(task_name):
    if USE_TOOLS_SETTING in ("1", "true", "yes", "on"): return True
    if USE_TOOLS_SETTING in ("0", "false", "no", "off"): return False
    return task_name == "polymorphic-zero-day"  # "auto" mode
```

By default, tools are only used automatically for the hard task.

---

### 10.4 LLM Action Extraction

```python
def get_action(client, obs):
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": sys_msg},
            {"role": "user",   "content": user_msg},
        ],
        temperature=0.1,
        max_tokens=300,
        stream=False,
    )
    text = resp.choices[0].message.content.strip()

    # Strip markdown fences if LLM wraps in ```json ... ```
    if "```" in text:
        for part in text.split("```"):
            if "{" in part:
                text = part.strip().lstrip("json").strip()
                break

    return json.loads(text)
```

Temperature 0.1 for near-deterministic output. JSON parsing with markdown fence stripping handles common LLM output formatting. On failure, falls back to a safe phase-correct action (Phase 1: `brute_force/monitor`, Phase 2: `monitor/auth_service`).

---

## PART 11 — Client (`client.py`)

### 11.1 `AdaptshieldEnv`

The Python client wraps OpenEnv's `EnvClient` with AdaptShield-specific types:

```python
class AdaptshieldEnv(EnvClient[AdaptShieldAction, AdaptShieldObservation, State]):
```

It maintains a **persistent WebSocket connection** to the server, enabling multi-step interactions with lower latency than HTTP round-trips.

**Payload serialization:**
```python
def _step_payload(self, action: AdaptShieldAction) -> Dict:
    return action.model_dump(
        mode="json",
        exclude_none=True,
        exclude_defaults=True,
    )
```

`exclude_none=True` + `exclude_defaults=True` ensures only populated fields are sent — Phase 1 payloads don't include `action=null`, Phase 2 payloads don't include `threat_type=null`. This prevents validation errors on the server.

**Usage pattern:**
```python
with AdaptshieldEnv(base_url="http://localhost:7860") as client:
    result = client.reset()
    obs = result.observation
    result = client.step(AdaptShieldAction(
        threat_type="brute_force",
        confidence=0.9,
        target_node="auth_service",
        recommended_action="rate_limit",
    ))
    print(result.observation.phase)   # 2
```

---

## PART 12 — SOC Tool Helpers (`soc_tools.py`)

### 12.1 Heuristic Threat Classifier

```python
def classify_from_metrics(network_nodes):
    auth    = network_nodes.get("auth_service", {})
    payment = network_nodes.get("payment_service", {})
    database= network_nodes.get("database", {})
    gateway = network_nodes.get("api_gateway", {})

    if float(auth.get("error_rate", 0.0)) >= 0.10:       return "brute_force"
    if payment.get("status") == "suspicious" or float(payment.get("cpu", 0)) >= 55:
                                                           return "lateral_movement"
    if float(database.get("outbound_mb", 0)) >= 50:       return "exfiltration"
    if gateway.get("status") == "suspicious":             return "supply_chain"
    return "benign"
```

This rule-based classifier is used to decide which tools to call in the investigation phase. It's the same logic in `baseline.py` and `train_smoke.py`.

---

### 12.2 Tool Investigation Plans

```python
THREAT_TOOL_PLAN = {
    "brute_force":      [("log_search", "auth_service"), ("cmdb_lookup", "auth_service")],
    "lateral_movement": [("edr_status", "payment_service"), ("log_search", "payment_service")],
    "exfiltration":     [("log_search", "database"), ("edr_status", "database")],
    "supply_chain":     [("vuln_lookup", "api_gateway"), ("log_search", "api_gateway")],
    "benign":           [("cmdb_lookup", "api_gateway")],
}

FALLBACK_SWEEP = [
    ("edr_status", "payment_service"),
    ("log_search", "database"),
    ("vuln_lookup", "api_gateway"),
]
```

For each suspected threat, the tool plan prescribes exactly which tools to call on which nodes. The fallback sweep runs if the initial tools don't find attack indicators (low confidence result).

---

### 12.3 Evidence Fusion

```python
def _complete_evidence_fusion(call_tool, results):
    belief = infer_threat_from_tool_results(results)
    threat = belief["threat_type"]
    if threat == "benign":
        return
    called = {(result["tool"], result["node"]) for result in results}
    for tool_name, node in THREAT_TOOL_PLAN.get(threat, []):
        if (tool_name, node) not in called:
            results.append(call_tool(tool_name, node))
```

After initial investigation, if a threat is detected, `_complete_evidence_fusion` ensures the **full required tool pair** for that threat is called. This is what enables `tool_evidence_found = True` in the grader — both tools from `REQUIRED_TOOL_FUSION` must have been called.

---

### 12.4 Threat Inference from Tool Results

```python
def infer_threat_from_tool_results(results):
    text_blob = " ".join([result_summary + events + finding + process_note ...]).lower()

    if any(result.get("risk") == "critical") or "hash mismatch" in text_blob:
        return {"threat_type": "supply_chain", ...}
    if "sequential reads" in text_blob or "compressed archive" in text_blob:
        return {"threat_type": "exfiltration", ...}
    if any(result.get("beaconing")) or "service account" in text_blob:
        return {"threat_type": "lateral_movement", ...}
    if "failed logins" in text_blob or "password spray" in text_blob:
        return {"threat_type": "brute_force", ...}
    return {"threat_type": "benign", ...}
```

Simple keyword matching on the public tool result text — no LLM. The agent (or training harness) uses this to make a decision about additional tool calls.

---

## PART 13 — Rule Baseline (`baseline.py`)

### 13.1 Perfect Deterministic Policy

The baseline implements the optimal rule-based policy:

```python
POLICY = {
    "brute_force":      ("auth_service",    "rate_limit"),
    "lateral_movement": ("payment_service", "isolate"),
    "exfiltration":     ("database",        "honeypot"),
    "supply_chain":     ("api_gateway",     "patch"),
    "benign":           ("api_gateway",     "monitor"),
}

def phase1_payload(obs):
    threat_type = classify_from_metrics(obs.network_nodes)
    target_node, action = POLICY[threat_type]
    return {
        "threat_type": threat_type, "confidence": 0.90,
        "target_node": target_node, "recommended_action": action,
    }

def phase2_payload(obs):
    assessment = obs.phase1_assessment
    threat_type = assessment.get("threat_type", "benign")
    action = assessment.get("recommended_action") or POLICY[threat_type][1]
    target_node = assessment.get("target_node") or POLICY[threat_type][0]
    return {"action": action, "target_node": target_node}
```

**Why does the baseline NOT score 0.99?** Because:
1. `polymorphic-zero-day` has 15% benign noise turns — the heuristic classifier sometimes misclassifies them
2. The hard task's mid-episode strategy shift at turn 3 means the classifier reads shifted metrics
3. No tool evidence → receives `P2_UNVERIFIED` (+0.10) instead of `P2_OPTIMAL` (+0.39) on hard task
4. Mission alignment penalties apply on any suboptimal action

Baseline scores: `direct-triage=0.87`, `dual-pivot=0.76`, `polymorphic-zero-day=0.52`

---

## PART 14 — Testing (`tests/test_regression.py`)

### 14.1 Test Suite Structure

Four test classes covering critical invariants:

**`PackageRegressionTests`:**
- `test_package_import_exports_expected_symbols` — Verifies `__all__` in `adaptshield/__init__.py` contains `AdaptShieldAction`, `AdaptShieldObservation`, `AdaptshieldEnv`
- `test_server_app_imports_fastapi_instance` — Verifies `app` is a `FastAPI` instance (not None, not dict)

**`EnvironmentRegressionTests`:**
- `test_phase_flow_accepts_both_action_shapes` — Full two-phase turn, checks phase transitions, reward > 0.9 for optimal action, `score_breakdown` in metadata, `active_defenses` in metadata
- `test_client_payload_omits_empty_metadata_and_serializes_enums` — Verifies `exclude_none` serialization strips null fields and enum values serialize as strings not enum names
- `test_hard_task_records_verified_tool_evidence` — Calls `call_tool()` before acting, verifies `tool_verification_required = True`, `tool_evidence_found = True`, `reward >= 0.85`
- `test_prompt_bank_builds_phase_rows_without_gpu_deps` — Builds a prompt bank with curriculum, verifies both phases covered, hard task rows have `tool_calls >= 2`

---

### 14.2 Running Tests

```bash
# From repo root, using the package virtualenv:
adaptshield/.venv/bin/python -m unittest tests.test_regression -v

# Preflight gate (full validation suite):
ADAPTSHIELD_SEED=42 adaptshield/.venv/bin/python eval_tasks.py
ADAPTSHIELD_SEED=42 adaptshield/.venv/bin/python baseline.py --task polymorphic-zero-day --replay
adaptshield/.venv/bin/python -m unittest tests.test_regression -v
adaptshield/.venv/bin/python smoke_test.py
cd adaptshield && adaptshield/.venv/bin/openenv validate
```

---

## PART 15 — Training Visualization (`plot_training.py`)

```python
def plot(path, output):
    episodes, scores, label, stages = load_scores(path)
    window = max(1, min(10, len(scores) // 5))
    smoothed = moving_average(scores, window)

    ax.plot(episodes, scores, alpha=0.35, label="raw score")
    ax.plot(episodes, smoothed, linewidth=2.5, label=f"{window}-episode avg")

    # Mark curriculum stage boundaries with vertical lines
    for episode, stage in stage_boundaries(episodes, stages):
        ax.axvline(episode, color="#c44e52", linestyle="--", alpha=0.45)
        ax.text(episode, 0.04, stage.replace("curriculum:", ""), ...)
```

Supports both CSV (smoke training output) and JSON (GPU training `metrics.json`). Stage boundaries from curriculum training appear as red dashed vertical lines — you can visually see where the agent transitions from easy to medium to hard task.

---

*End of Part 5. Continue to Part 6: Strengths, Limitations, Pitching & Research Directions.*
