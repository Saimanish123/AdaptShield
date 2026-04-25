# AdaptShield Project Deep Dive

This document is a full technical walkthrough of the AdaptShield project.
It is written as an internal preparation guide so you can understand:

- what the project is trying to do
- why it is structured the way it is
- how the environment works end to end
- how each file fits into the system
- how the hard task was hardened for world modeling
- how training, inference, baselines, and evaluation are connected
- what has already been validated
- what is still a limitation or future step

The goal is to make this useful both as a study document and as a briefing note
before the hackathon.

## 1. Project In One Sentence

AdaptShield is a two-phase cybersecurity environment for OpenEnv where an agent
must:

1. classify an incident from partial and evolving evidence
2. hand off that assessment to a second decision phase
3. execute a defense
4. reason across changing attacker state, tool outputs, and mission tradeoffs

On the hardest task, high score requires tool-mediated investigation and
context-dependent decision making rather than one-step pattern matching.

## 2. Core Idea

Most simple cyber environments reduce to:

- read current metrics
- map metrics to attack label
- map label to defense

That is too shallow for world modeling.

AdaptShield tries to go beyond that by combining:

- a two-phase agent workflow
- attacker stage progression across turns
- task difficulty progression
- partial observability
- hidden attacker state
- stateful investigative tools
- operational and mission tradeoffs

The intended story is not "classify the alert."

The intended story is:

- observe an ambiguous incident
- investigate with tools
- build a belief about attacker state
- choose a response
- observe consequences
- adapt on later turns

## 3. Theme Alignment

AdaptShield is aimed at OpenEnv Theme #3: World Modeling / Professional Tasks.

Why it fits:

- The environment is dynamic, not static.
- The model interacts with tools and APIs, not just one observation.
- Hidden state exists and changes over time.
- The model needs to update belief based on outcomes.
- The hard task cannot be solved well by a no-tool agent.
- The hard task is now intentionally harder for rule-based tool agents too.

Important design choice:

- We use local, deterministic enterprise-style APIs rather than live external
  internet APIs.

This is intentional because:

- it is reproducible
- it is trainable
- it is cheaper
- it is judge-safe
- it avoids auth, latency, and outage problems

## 4. High-Level Architecture

The project has five main layers:

1. Environment layer
   - implements the two-phase cyber environment
   - owns turn state, rewards, task logic, hidden state, replay, and metadata

2. Attacker layer
   - generates the current incident world
   - chooses active strategy and stage
   - creates alerts and network metrics

3. Grader layer
   - computes deterministic reward
   - models optimal, heavy, wrong, catastrophic, and unverified responses
   - adds mission and operational impact

4. Tool/API layer
   - exposes stateful SOC-style investigative tools
   - returns partial evidence, not ground-truth labels

5. Agent/training/eval layer
   - baselines
   - inference script
   - smoke trainer
   - GPU trainer
   - optional GRPO path

## 5. Two-Phase Design

This is one of the most important architectural decisions in the project.

### Phase 1: Threat Analyst

The agent sees:

- `network_nodes`
- `active_alerts`
- `attack_stage`
- recent history
- mission context
- optional tool evidence

It must output:

- `threat_type`
- `confidence`
- `target_node`
- `recommended_action`
- optional `reasoning`

### Phase 2: Tactical Executor

The agent does not see raw network state.

It sees only:

- the analyst handoff
- recent history
- optional tool trace for the current turn
- mission context

It must output:

- `action`
- `target_node`
- optional `reasoning`

### Why This Matters

This split creates a richer problem than one monolithic action:

- classification accuracy matters
- communication quality matters
- execution quality matters
- handoff degradation can matter
- the second phase has to act under information loss

That is much closer to real enterprise security workflows.

## 6. Difficulty Ladder

There are three tasks.

### `direct-triage`

Purpose:

- teach basic classification and response mapping

Properties:

- fixed simple patterns
- no mandatory tool use
- short episode
- should be the easiest task

### `dual-pivot`

Purpose:

- test adaptation when the attacker pivots mid-episode

Properties:

- attacker shifts from earlier behavior to new behavior
- tool use can help, but is not required for high score
- medium difficulty

### `polymorphic-zero-day`

Purpose:

- force genuine world modeling and investigation

Properties:

- all attack families are possible
- noise injection exists
- attacker can pivot
- hidden foothold state matters
- hard task requires multi-tool evidence for full credit
- public tool evidence is intentionally less one-to-one now

This task is the benchmark's main innovation target.

## 7. Attacker Model

File:

- `adaptshield/server/attacker.py`

The attacker is scripted and deterministic under seed.
It is not an LLM.

### What It Controls

- active attack strategy
- current attack stage
- signal generation
- benign noise injection
- mid-episode strategy shift

### Strategies

- `brute_force`
- `lateral_movement`
- `exfiltration`
- `supply_chain`

### Stages

- `recon`
- `exploit`
- `exfiltration`

If the agent does not act correctly, the attacker escalates stage.

### Difficulty-Specific Behavior

- `direct-triage`: no meaningful pivot
- `dual-pivot`: clean pivot after a few turns
- `polymorphic-zero-day`: more complex shift behavior and noise

### Seeding

`ADAPTSHIELD_SEED` is supported so the world is reproducible across runs.

This is important for:

- debugging
- evaluation consistency
- training stability

## 8. Scenario Builder

File:

- `adaptshield/server/scenarios.py`

This file builds the actual observations that the agent sees.

### Key Responsibilities

- define task configs
- define mission profiles
- build Phase 1 observations
- build Phase 2 observations
- inject system prompt context

### Important Detail

Phase 2 observations explicitly zero out:

- `network_nodes`
- `active_alerts`

This is how Phase 2 blindness is enforced.

### Mission Context

Each task has a mission profile with:

- `mission_id`
- `primary_asset`
- `sla_priority`
- `risk_tolerance`
- `objective`

This is used to make the benchmark more realistic than "just stop the attack."

## 9. Environment Core

File:

- `adaptshield/server/adaptshield_environment.py`

This is the center of the project.

### Responsibilities

- reset and step logic
- phase transition management
- reward handling
- attacker advancement
- defense persistence
- hidden foothold state
- tool recording
- replay generation
- metadata construction
- error-safe observations

### Environment State Stored Internally

- current task
- current turn
- current phase
- reward history
- last reward
- conversation history
- phase 1 output
- current turn config
- consecutive wrong actions
- active defenses with TTL
- foothold state
- tool trace
- verified evidence per turn
- full tool results per turn
- final replay rows

### Reset Flow

1. reset state
2. reset attacker episode
3. build initial turn config
4. build Phase 1 observation
5. attach metadata

### Step Flow

#### If current phase is 1

1. parse analyst payload safely
2. save raw grading copy
3. optionally degrade handoff on the hard task
4. build Phase 2 observation
5. keep `normalized_score` present

#### If current phase is 2

1. parse executor action safely
2. call `grade_step(...)`
3. clamp reward
4. register active defense
5. update foothold state if needed
6. update history and replay
7. advance attacker
8. decay defense TTLs
9. move back to Phase 1 or finish episode
10. always include `normalized_score`

### Error Handling

`step()` is wrapped so it returns a safe observation even on exceptions.
This is important for robustness and evaluation stability.

## 10. Hidden State: Foothold

One of the key hard-task innovations is the foothold mechanic.

### What It Means

Foothold means the attacker has already gained persistence or spread enough
that the same observed activity now requires a different response.

### Why It Exists

Without hidden state, the hard task collapses into:

- detect current strategy
- choose fixed response

With foothold, the agent has to reason:

- did I miss containment earlier?
- did the attacker already gain persistence?
- is direct isolation still the right move?

### Current Rule

On the hard task:

- if containment was missed and foothold is established
- later lateral-movement-like behavior can require `honeypot` instead of
  `isolate`

### Important Security Improvement

Live observation metadata no longer directly exposes `foothold_established`.
That hidden state is still reflected indirectly through world consequences and
tool outputs, but it is not handed to the agent as ground truth each turn.

## 11. Handoff Degradation

File:

- `adaptshield/server/adaptshield_environment.py`

The hard task can degrade the analyst handoff later in the episode.

This means:

- the analyst may become less confident
- the recommended action can become too passive
- the executor must decide whether the handoff is now unreliable

This improves realism because real incident handoffs are not always clean.

## 12. Active Defenses

The environment tracks active defenses like:

- `rate_limit`
- `isolate`
- `honeypot`
- `patch`

Each has a TTL and side effect.

Why this matters:

- actions persist across turns
- the world is not memoryless
- tools like EDR can reflect active controls

## 13. Grader

File:

- `adaptshield/server/grader.py`

The grader is deterministic and fast.
It does not use NLP and does not use LLM-as-judge logic.

### What It Scores

- Phase 1 threat type accuracy
- Phase 1 target accuracy
- Phase 2 action quality
- catastrophic outcomes
- false positives
- escalation penalties
- mission alignment
- operational disruption
- tool verification requirement on hard task

### Reward Categories

- optimal
- heavy-handed
- wrong
- false positive
- unverified correct action
- catastrophic

### Why Deterministic Grading Matters

- reproducibility
- no hallucinated judging
- faster training
- easier debugging

## 14. Mission and Operational Modeling

The grader includes:

- asset criticality
- service dependencies
- disruption cost by action
- mission alignment adjustment

This is intentionally lightweight.

The project is not trying to be a huge economic simulator.
It is trying to add realistic tradeoffs without making rewards too noisy.

## 15. Tool/API Layer

Files:

- `adaptshield/server/app.py`
- `adaptshield/server/adaptshield_environment.py`

### Endpoints

Core OpenEnv:

- `/reset`
- `/step`

Additional SOC API layer:

- `/soc/reset`
- `/soc/step`
- `/tools/log_search`
- `/tools/cmdb_lookup`
- `/tools/edr_status`
- `/tools/vuln_lookup`

### Important Architecture Decision

The OpenEnv contract remains unchanged.
The tool layer is additive.

That means:

- OpenEnv validation still works
- baseline and smoke paths still work
- stronger agents can choose to use the SOC tool surface

### Tool Intent

#### `log_search`

Returns recent log-like evidence about a node.

Used for:

- authentication anomalies
- east-west movement clues
- data-access clues
- release/update anomalies

#### `cmdb_lookup`

Returns service context:

- owner
- criticality
- dependencies
- safe actions

Used for:

- blast radius reasoning
- action tradeoff reasoning

#### `edr_status`

Returns endpoint status:

- containment state
- persistence
- beaconing
- process notes
- active controls

Used for:

- hidden state inference
- containment verification

#### `vuln_lookup`

Returns software/advisory context.

Used for:

- supply chain reasoning
- patch vs monitor vs isolate decisions

## 16. Public vs Internal Tool Data

This distinction is critical.

### Internal Tool Data

The environment internally tracks:

- `evidence_type`
- `verified`
- exact evidence alignment

This is used by the grader.

### Public Tool Data

The agent only sees observable facts.

We explicitly hide:

- `evidence_type`
- `verified`

This is what prevents the tool layer from becoming an answer key.

## 17. Hard-Task Hardening

The project went through several hardening passes.

### Earlier Problem

The tool-aware baseline became too strong because the tool outputs were too
cleanly separable and the rule logic could map them directly to actions.

### What Was Changed

On `polymorphic-zero-day`:

- evidence wording became less one-to-one
- overlapping signals were introduced
- some previously explicit keywords were softened
- live hidden state exposure was removed
- full credit still requires investigation

### Current Goal

The hard task should satisfy:

- no-tool agents fail clearly
- naive tool agents improve but do not dominate
- trained reasoning agents can outperform both

## 18. Baselines

### `baseline.py`

This is the no-tool rule baseline.

It:

- classifies threat from raw metrics
- outputs analyst handoff
- mirrors that handoff into executor action

Purpose:

- provide the difficulty staircase
- verify environment validity

### `tool_baseline.py`

This is the tool-aware rule baseline.

It now:

- queries tools in Phase 1
- builds a belief from observable fields
- uses that belief in Phase 1 and Phase 2

Purpose:

- demonstrate the value of investigation
- provide a lower-bound tool-using benchmark

Important:

- it is still intentionally naive
- it is not meant to be the final champion agent

## 19. Evaluation Scripts

### `eval_tasks.py`

Runs the no-tool baseline over all tasks and prints:

- task
- score
- steps
- `normalized_score` presence
- pass/fail
- staircase pass/fail

### `smoke_test.py`

Minimal repo-root validation:

- imports
- app presence
- phase transition
- basic reward

### `tests/test_regression.py`

Regression tests for:

- package exports
- FastAPI app import
- phase flow
- client serialization
- hidden tool-label behavior
- prompt bank generation

## 20. Inference Path

File:

- `inference.py`

This is the evaluator-style script that emits:

- `[START]`
- `[STEP]`
- `[END]`

### Two Modes

#### OpenEnv path

Uses the standard client against `/reset` and `/step`.

#### SOC path

Uses `/soc/reset`, `/tools/*`, and `/soc/step`.

This is used when tool-aware behavior is enabled, especially on the hard task.

### Important Design Point

Credentials are read from environment variables.
No live token is hardcoded in the file.

## 21. Training Paths

There are three training-related paths in the repo.

### `train_smoke.py`

A dependency-free tabular smoke trainer.

Purpose:

- verify trainability
- verify repeated env interaction
- produce a quick staircase-like curve

This is not the real hackathon training path.

### `train.py` Policy-Gradient Path

This is a fallback GPU trainer using:

- Unsloth
- LoRA
- a simple policy-gradient-like update loop

Purpose:

- safe onsite fallback
- simpler than full RLHF/GRPO stack

### `train.py` GRPO Path

This is the more advanced path.

It:

- builds an env-derived prompt bank from actual reference rollouts
- uses a GRPO/TRL-style trainer when dependencies are available
- evaluates the trained model back on the environment

Important reality:

- the GRPO path is implemented
- but it still needs a real GPU dependency/runtime validation in the target setup

That is normal at this stage.

## 22. Prompt Bank Design

The GRPO path does not train on arbitrary canned prompts.

It creates prompts from real env states:

- observation
- history
- tool results
- phase
- task
- expected action labels derived from env truth

This is important because it keeps the training data aligned with the actual
benchmark.

## 23. Plotting and Metrics

File:

- `plot_training.py`

The trainer writes `metrics.json`, including:

- trainer type
- whether tools were used
- curriculum stages
- training rows
- evaluation rows
- prompt bank size when relevant

This makes it easier to:

- show reward curves
- compare runs
- generate judge-facing plots

## 24. File Structure

Top-level files:

- `baseline.py`
- `eval_tasks.py`
- `inference.py`
- `plot_training.py`
- `PROJECT_DEEP_DIVE.md`
- `smoke_test.py`
- `soc_tools.py`
- `tool_baseline.py`
- `train.py`
- `train_smoke.py`
- `TRAINING.md`
- `WORLD_MODELING.md`

Package root:

- `adaptshield/__init__.py`
- `adaptshield/client.py`
- `adaptshield/models.py`
- `adaptshield/openenv.yaml`
- `adaptshield/README.md`
- `adaptshield/pyproject.toml`

Server package:

- `adaptshield/server/app.py`
- `adaptshield/server/adaptshield_environment.py`
- `adaptshield/server/attacker.py`
- `adaptshield/server/grader.py`
- `adaptshield/server/scenarios.py`

Tests:

- `tests/test_regression.py`

## 25. Most Important Files to Study First

If you are short on time, study these first:

1. `adaptshield/server/adaptshield_environment.py`
2. `adaptshield/server/grader.py`
3. `adaptshield/server/attacker.py`
4. `adaptshield/server/scenarios.py`
5. `inference.py`
6. `train.py`
7. `soc_tools.py`

## 26. Design Choices That Matter Most

### Separate Phase 1 and Phase 2 action semantics

This avoids payload confusion and makes the benchmark richer.

### Deterministic grader

This keeps the benchmark stable and trainable.

### Additive tool layer

This preserved OpenEnv compliance while improving world-modeling alignment.

### Hidden state with indirect evidence

This is essential for making the hard task more than a classification game.

### Bounded penalties and dense rewards

This helps training remain feasible.

## 27. What Has Been Validated

At the time of writing, the following have been checked locally:

- OpenEnv validation passes
- smoke test passes
- regression tests pass
- no-tool task evaluation passes
- difficulty staircase passes
- tool-aware smoke flow passes
- public tools do not expose hidden `verified` / `evidence_type`
- Phase 2 is genuinely blind to raw state

## 28. Current Benchmark Shape

Representative recent local behavior:

- no-tool easy: high
- no-tool medium: lower
- no-tool hard: low
- naive tool hard: improved, but not dominant

Exact values vary if seed is not fixed.
When comparing numbers, always prefer fixed `ADAPTSHIELD_SEED`.

## 29. Known Limitations

These are real limitations, not marketing omissions.

### 1. Real GPU training proof is still pending

The environment and training code are in place, but a full serious 7B run has
not yet been proven in the final target GPU setup.

### 2. Tool-aware rule baselines are still handcrafted

That is fine for baselines, but they are not the end goal.

### 3. Logs are simulated, not sourced from real enterprise systems

This is acceptable for a benchmark, but should be described honestly.

### 4. The hard task is now meaningfully harder, but the final truth will come
from model training results

The benchmark is only truly vindicated if a model learns above the naive
baselines.

## 30. Why This Project Is Stronger Now Than The Earlier Prototype

Earlier it was closer to:

- "a cyber RL prototype with three tasks"

Now it is much closer to:

- "a partially observable cyber incident-response world with two-phase agency,
  persistent attacker state, stateful investigative APIs, mission tradeoffs,
  and trainable reward structure"

That is a materially better story.

## 31. Suggested Study Order For You

If you want to prepare well, read in this order:

1. this document
2. `WORLD_MODELING.md`
3. `TRAINING.md`
4. `adaptshield/server/adaptshield_environment.py`
5. `adaptshield/server/grader.py`
6. `adaptshield/server/attacker.py`
7. `inference.py`
8. `train.py`

After that, run:

```bash
python eval_tasks.py
python smoke_test.py
python tool_baseline.py --task polymorphic-zero-day
python train.py --smoke --task all --curriculum --use-tools --episodes 6
```

## 32. Final Mental Model

If you remember only one mental model, remember this:

AdaptShield is not trying to be a giant real SOC.
It is trying to be a clean, trainable, world-modeling benchmark where:

- the analyst and executor are separated
- the world changes over time
- hidden attacker state matters
- tools reveal evidence, not answers
- the hardest task rewards cross-turn reasoning
- training is still possible because the structure is disciplined

That is the essence of the project.
