# AdaptShield World-Modeling Layer

AdaptShield now includes a local, stateful SOC API surface for Theme #3.1
Professional Tasks / World Modeling.

The core OpenEnv contract is unchanged:

```text
/reset
/step
same action schema
same three tasks
same normalized_score metadata
```

The added layer exposes investigative tools:

```text
POST /soc/reset
POST /soc/step
POST /tools/log_search
POST /tools/cmdb_lookup
POST /tools/edr_status
POST /tools/vuln_lookup
```

The tools reveal partial evidence, not answers. Public tool responses do not
include grader labels like `verified` or `evidence_type`; agents must infer from
observable facts such as failed-login clusters, beaconing, hash mismatches,
service dependencies, and database read patterns. Tool responses depend on the
current turn, attacker stage, foothold state, active defenses, mission profile,
and previous actions.

On `polymorphic-zero-day`, high score now requires multi-tool evidence fusion.
A correct action without enough investigation still works, but receives
`Unverified correct action` instead of full optimal credit. This makes the hard
task a partially observable workflow instead of a static classification problem.

The intended agent loop is:

```text
observe ambiguous incident snapshot
query SOC tools
update belief about attacker state
choose defensive action
observe consequences
repeat
```

Useful comparison:

```bash
ADAPTSHIELD_SEED=42 python eval_tasks.py
ADAPTSHIELD_SEED=42 python tool_baseline.py --task polymorphic-zero-day --replay
```

Tool-aware inference and GPU training:

```bash
ADAPTSHIELD_TASK=polymorphic-zero-day ADAPTSHIELD_USE_TOOLS=auto python inference.py
python train.py --task all --curriculum --use-tools --model 7b --episodes 300
```

Expected behavior under the fixed seed:

```text
no-tool hard baseline:      ~0.37, still valid
naive tool hard baseline:   ~0.66, tools help but do not solve it
trained tool-aware LLM:     target higher through evidence fusion and reasoning
```
