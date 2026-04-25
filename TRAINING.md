# AdaptShield Training Guide

This repo has three training paths:

1. Local smoke training, which is dependency-free and verifies the environment can run repeated learning loops.
2. Onsite GPU policy-gradient training, which is the safe fallback when TRL/GRPO dependencies are unavailable.
3. Onsite GPU GRPO training, which builds an env-derived prompt bank and trains with TRL when the full stack is available.

## Local Smoke Training

Run this before spending credits:

```bash
cd /Users/manish/adaptshield
ADAPTSHIELD_SEED=42 /Users/manish/adaptshield/adaptshield/.venv/bin/python train_smoke.py --task direct-triage --episodes 30
```

This writes:

```text
training_runs/train_smoke.csv
```

Optional plot:

```bash
/Users/manish/adaptshield/adaptshield/.venv/bin/python plot_training.py \
  --input training_runs/train_smoke.csv \
  --output training_runs/reward_curve.png
```

If `matplotlib` is not installed, the plotter prints the score trend instead of failing.

For an all-task stability stress test:

```bash
ADAPTSHIELD_SEED=42 /Users/manish/adaptshield/adaptshield/.venv/bin/python train_smoke.py --task all --episodes 30
```

For the judge-facing curriculum path, use the training harness. This starts on the easy task, moves to the pivot task, then finishes on the hard contextual task:

```bash
ADAPTSHIELD_SEED=42 /Users/manish/adaptshield/adaptshield/.venv/bin/python train.py \
  --smoke \
  --curriculum \
  --episodes 30 \
  --output training_runs/curriculum-smoke \
  --plot
```

## GPU Training

Use this onsite after credits/GPU are available. `--trainer auto` prefers GRPO and falls back safely:

```bash
cd /Users/manish/adaptshield
pip install unsloth trl accelerate bitsandbytes matplotlib
export HF_TOKEN=your_huggingface_token
ADAPTSHIELD_SEED=42 python train.py \
  --task all \
  --curriculum \
  --trainer auto \
  --model 1.5b \
  --episodes 60 \
  --output checkpoints/adaptshield-qwen25-15b \
  --plot
```

For a cheaper first GPU test:

```bash
python train.py --task direct-triage --model 0.5b --episodes 5 --output checkpoints/smoke-gpu --plot
```

For the main onsite run, use curriculum on the strongest model your credits/time can support:

```bash
ADAPTSHIELD_SEED=42 python train.py \
  --task all \
  --curriculum \
  --use-tools \
  --trainer auto \
  --model 7b \
  --episodes 300 \
  --output checkpoints/adaptshield-qwen25-7b-curriculum \
  --plot
```

If you want to force the real TRL/GRPO path explicitly:

```bash
ADAPTSHIELD_SEED=42 python train.py \
  --task all \
  --curriculum \
  --use-tools \
  --trainer grpo \
  --model 7b \
  --prompt-bank-episodes 36 \
  --grpo-epochs 2 \
  --num-generations 4 \
  --per-device-batch-size 1 \
  --gradient-accumulation-steps 4 \
  --eval-episodes 2 \
  --output checkpoints/adaptshield-qwen25-7b-grpo \
  --plot
```

`metrics.json` now includes:

```text
trainer
prompt_bank_size
rows               # training curve rows
evaluation_rows    # online post-train env scores by task
```

For a no-GPU fallback:

```bash
python train.py --smoke --task all --curriculum --episodes 30 --plot
```

For a no-GPU rehearsal of the tool-aware world-modeling path:

```bash
python train.py --smoke --task all --curriculum --use-tools --episodes 30 --plot
```

## Tool-Aware Inference

For `polymorphic-zero-day`, `inference.py` uses the SOC tool API automatically
when `ADAPTSHIELD_USE_TOOLS=auto` or unset. This makes the model see tool
evidence before acting on the hard partially observable task.

```bash
cd /Users/manish/adaptshield/adaptshield
ADAPTSHIELD_SEED=42 python -m server.app --port 7860
```

In another terminal:

```bash
cd /Users/manish/adaptshield
ADAPTSHIELD_TASK=polymorphic-zero-day \
ADAPTSHIELD_USE_TOOLS=auto \
python inference.py
```

You can force the old no-tool OpenEnv path with:

```bash
ADAPTSHIELD_USE_TOOLS=0 ADAPTSHIELD_TASK=polymorphic-zero-day python inference.py
```

## Outputs

GPU training writes:

```text
checkpoints/<run-name>/best
checkpoints/<run-name>/final
checkpoints/<run-name>/metrics.json
```

Plot GPU metrics:

```bash
python plot_training.py \
  --input checkpoints/adaptshield-qwen25-15b/metrics.json \
  --output checkpoints/adaptshield-qwen25-15b/reward_curve.png
```

For tool-aware GPU metrics, use the 7B curriculum output path:

```bash
python plot_training.py \
  --input checkpoints/adaptshield-qwen25-7b-curriculum/metrics.json \
  --output checkpoints/adaptshield-qwen25-7b-curriculum/reward_curve.png
```

## Preflight Gate

Run this before any serious training:

```bash
ADAPTSHIELD_SEED=42 /Users/manish/adaptshield/adaptshield/.venv/bin/python eval_tasks.py
ADAPTSHIELD_SEED=42 /Users/manish/adaptshield/adaptshield/.venv/bin/python baseline.py --task polymorphic-zero-day --replay
/Users/manish/adaptshield/adaptshield/.venv/bin/python -m unittest tests.test_regression -v
/Users/manish/adaptshield/adaptshield/.venv/bin/python smoke_test.py
cd /Users/manish/adaptshield/adaptshield && /Users/manish/adaptshield/adaptshield/.venv/bin/openenv validate
```
