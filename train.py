#!/usr/bin/env python3
"""AdaptShield onsite GPU training harness with safe local fallback."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


REPO_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = REPO_ROOT / "adaptshield"

if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from models import AdaptShieldAction
from server.adaptshield_environment import AdaptShieldEnvironment
from train_smoke import TASKS, run_smoke_training
from plot_training import plot as plot_training_output
from soc_tools import attach_tool_results, investigate_local_with_depth, summarize_tool_results


MODEL_CHOICES = {
    "0.5b": "unsloth/Qwen2.5-0.5B-Instruct",
    "1.5b": "unsloth/Qwen2.5-1.5B-Instruct",
    "3b": "unsloth/Qwen2.5-3B-Instruct",
    "7b": "unsloth/Qwen2.5-7B-Instruct",
}
DEFAULT_MODEL = "1.5b"
MAX_SEQ_LEN = 2048
LORA_RANK = 16

CURRICULUM_STAGES = [
    ("direct-triage", 0.30),
    ("dual-pivot", 0.40),
    ("polymorphic-zero-day", 0.30),
]


PHASE1_SYS = """You are a Threat Analyst for a 4-node enterprise network.
Analyze SIEM metrics and alerts. Identify the threat.

Attack strategies: brute_force, lateral_movement, exfiltration, supply_chain, benign
Nodes: auth_service, payment_service, database, api_gateway
Actions: rate_limit, isolate, honeypot, patch, monitor
If SOC tool evidence is provided, use it to update your belief before classifying.

Respond ONLY with valid JSON:
{"threat_type":"...","confidence":0.0,"target_node":"...","recommended_action":"...","reasoning":"..."}"""


PHASE2_SYS = """You are a Tactical Executor. Act only on the analyst handoff.
You cannot see raw network data in Phase 2.
Use the analyst handoff plus any SOC tool trace from this turn.

Actions: rate_limit, isolate, honeypot, patch, monitor
Nodes: auth_service, payment_service, database, api_gateway

Respond ONLY with valid JSON:
{"action":"...","target_node":"...","reasoning":"..."}"""


def obs_to_dict(obs: Any) -> Dict[str, Any]:
    if hasattr(obs, "model_dump"):
        return obs.model_dump(mode="json")
    return dict(obs)


def make_phase1_prompt(obs: Dict[str, Any]) -> str:
    return "\n".join([
        "Network nodes:",
        json.dumps(obs.get("network_nodes", {}), indent=2),
        "",
        "Active alerts:",
        "\n".join(obs.get("active_alerts", [])),
        "",
        "SOC tool evidence:",
        summarize_tool_results(obs.get("tool_results", [])),
        "",
        "Recent history:",
        json.dumps(obs.get("history", [])[-3:], indent=2),
        "",
        "Classify the threat:",
    ])


def make_phase2_prompt(obs: Dict[str, Any]) -> str:
    metadata = obs.get("metadata", {}) if isinstance(obs.get("metadata", {}), dict) else {}
    current_turn = int(obs.get("turn", 0) or 0)
    tool_trace = [
        row for row in metadata.get("tool_trace", [])
        if int(row.get("turn", -1)) == current_turn
    ]
    return "\n".join([
        "Threat assessment from analyst:",
        json.dumps(obs.get("phase1_assessment", {}), indent=2),
        "",
        "SOC tool trace for this turn:",
        json.dumps(tool_trace, indent=2),
        "",
        "Choose the defensive action:",
    ])


def build_messages(obs: Dict[str, Any]) -> List[Dict[str, str]]:
    if int(obs.get("phase", 1)) == 1:
        return [
            {"role": "system", "content": PHASE1_SYS},
            {"role": "user", "content": make_phase1_prompt(obs)},
        ]
    return [
        {"role": "system", "content": PHASE2_SYS},
        {"role": "user", "content": make_phase2_prompt(obs)},
    ]


def task_for_episode(
    episode: int,
    total_episodes: int,
    selected_task: str,
    curriculum: bool,
) -> Tuple[str, str]:
    if not curriculum:
        if selected_task == "all":
            task = TASKS[(episode - 1) % len(TASKS)]
            return task, "round_robin"
        return selected_task, "fixed"

    progress = episode / max(1, total_episodes)
    cumulative = 0.0
    for task, fraction in CURRICULUM_STAGES:
        cumulative += fraction
        if progress <= cumulative:
            return task, f"curriculum:{task}"
    return CURRICULUM_STAGES[-1][0], f"curriculum:{CURRICULUM_STAGES[-1][0]}"


def save_metrics(
    output_dir: Path,
    rows: List[Dict[str, Any]],
    model_name: str,
    episodes: int,
    curriculum: bool,
    use_tools: bool,
    trainer: str = "pg",
    evaluation_rows: List[Dict[str, Any]] | None = None,
    prompt_bank_size: int = 0,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_score = max((float(row["score"]) for row in rows), default=0.0)
    metrics_path = output_dir / "metrics.json"
    payload = {
        "model": model_name,
        "episodes": episodes,
        "curriculum": curriculum,
        "curriculum_stages": CURRICULUM_STAGES,
        "use_tools": use_tools,
        "trainer": trainer,
        "rows": rows,
        "best_score": best_score,
    }
    if evaluation_rows is not None:
        payload["evaluation_rows"] = evaluation_rows
    if prompt_bank_size:
        payload["prompt_bank_size"] = prompt_bank_size
    metrics_path.write_text(json.dumps(payload, indent=2))
    return metrics_path


def maybe_plot(metrics_path: Path, output_dir: Path) -> None:
    try:
        plot_training_output(metrics_path, output_dir / "reward_curve.png")
    except Exception as exc:
        print(f"Plot generation skipped: {exc}")


def parse_response(text: str, phase: int) -> Dict[str, Any]:
    """Parse model JSON. Invalid output becomes a safe phase-correct action."""
    if "```" in text:
        for part in text.split("```"):
            if "{" in part:
                text = part.strip().removeprefix("json").strip()
                break

    try:
        parsed = json.loads(text)
        if phase == 1:
            return {
                "threat_type": str(parsed.get("threat_type", "brute_force")),
                "confidence": float(parsed.get("confidence", 0.5)),
                "target_node": str(parsed.get("target_node", "auth_service")),
                "recommended_action": str(parsed.get("recommended_action", "monitor")),
                "reasoning": str(parsed.get("reasoning", "")),
            }
        return {
            "action": str(parsed.get("action", "monitor")),
            "target_node": str(parsed.get("target_node", "auth_service")),
            "reasoning": str(parsed.get("reasoning", "")),
        }
    except Exception:
        if phase == 1:
            return {
                "threat_type": "brute_force",
                "confidence": 0.5,
                "target_node": "auth_service",
                "recommended_action": "monitor",
                "reasoning": "parse_error",
            }
        return {
            "action": "monitor",
            "target_node": "auth_service",
            "reasoning": "parse_error",
        }


def render_messages(messages: List[Dict[str, str]], tokenizer: Any | None = None) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n\n".join(
        f"{message.get('role', 'user').upper()}:\n{message.get('content', '')}"
        for message in messages
    )


def generate_response(model: Any, tokenizer: Any, messages: List[Dict[str, str]]) -> Tuple[str, str]:
    import torch

    prompt = render_messages(messages, tokenizer=tokenizer)
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=220,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_ids = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
    return prompt, response


def _current_reference(env: AdaptShieldEnvironment) -> Dict[str, Any]:
    turn_config = dict(getattr(env, "_turn_config", {}) or {})
    is_benign = bool(turn_config.get("is_benign", False))
    threat_type = "benign" if is_benign else str(turn_config.get("strategy", "benign"))
    target_node = str(turn_config.get("correct_target", "auth_service"))
    expected_action = str(turn_config.get("correct_action", "monitor"))
    return {
        "threat_type": threat_type,
        "target_node": target_node,
        "expected_action": expected_action,
        "stage": str(turn_config.get("attack_stage", getattr(env._attacker, "current_stage", lambda: "recon")())),
        "is_benign": is_benign,
    }


def _teacher_payload(phase: int, reference: Dict[str, Any]) -> Dict[str, Any]:
    if phase == 1:
        return {
            "threat_type": reference["threat_type"],
            "confidence": 0.92 if reference["threat_type"] != "benign" else 0.78,
            "target_node": reference["target_node"],
            "recommended_action": reference["expected_action"],
            "reasoning": "reference policy",
        }
    return {
        "action": reference["expected_action"],
        "target_node": reference["target_node"],
        "reasoning": "reference policy",
    }


def build_prompt_bank(
    tokenizer: Any | None,
    selected_task: str,
    curriculum: bool,
    rollout_episodes: int,
    max_steps: int,
    use_tools: bool,
    seed: int,
) -> List[Dict[str, Any]]:
    random.seed(seed)
    rows: List[Dict[str, Any]] = []
    for episode in range(1, rollout_episodes + 1):
        task, stage = task_for_episode(
            episode=episode,
            total_episodes=rollout_episodes,
            selected_task=selected_task,
            curriculum=curriculum,
        )
        env = AdaptShieldEnvironment(task_name=task)
        obs = env.reset()
        while not obs.done and len(rows) < rollout_episodes * max_steps * 2:
            phase = int(getattr(obs, "phase", 1))
            tool_results = investigate_local_with_depth(
                env,
                obs,
                use_tools=use_tools,
                thorough=True,
            )
            obs_dict = attach_tool_results(obs_to_dict(obs), tool_results)
            messages = build_messages(obs_dict)
            reference = _current_reference(env)
            rows.append({
                "prompt": render_messages(messages, tokenizer=tokenizer),
                "task": task,
                "stage": stage,
                "phase": phase,
                "turn": int(getattr(obs, "turn", 0) or 0),
                "attack_stage": reference["stage"],
                "expected_threat_type": reference["threat_type"],
                "expected_target_node": reference["target_node"],
                "expected_recommended_action": reference["expected_action"] if phase == 1 else "",
                "expected_action": reference["expected_action"] if phase == 2 else "",
                "tool_calls": len(tool_results),
                "history_length": len(obs_dict.get("history", [])),
            })
            obs = env.step(AdaptShieldAction(**_teacher_payload(phase, reference)))
            if len(rows) >= rollout_episodes * max_steps * 2:
                break
    return rows


def _completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        if "content" in completion:
            return str(completion.get("content", ""))
        if "text" in completion:
            return str(completion.get("text", ""))
    if isinstance(completion, list):
        parts = []
        for item in completion:
            if isinstance(item, dict):
                parts.append(str(item.get("content", item.get("text", ""))))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(completion)


def _phase1_reward(
    parsed: Dict[str, Any],
    expected_threat_type: str,
    expected_target_node: str,
    expected_recommended_action: str,
) -> float:
    reward = 0.08
    if parsed.get("threat_type") == expected_threat_type:
        reward += 0.36
    if parsed.get("target_node") == expected_target_node:
        reward += 0.20
    if parsed.get("recommended_action") == expected_recommended_action:
        reward += 0.18
    try:
        confidence = float(parsed.get("confidence", 0.5))
    except Exception:
        confidence = 0.5
    if 0.0 <= confidence <= 1.0:
        reward += 0.05
    if parsed.get("threat_type") == expected_threat_type and confidence >= 0.65:
        reward += 0.06
    elif parsed.get("threat_type") != expected_threat_type and confidence >= 0.80:
        reward -= 0.05
    if parsed.get("recommended_action") == "monitor" and expected_threat_type != "benign":
        reward -= 0.05
    return max(0.01, min(0.99, round(reward, 2)))


def _phase2_reward(
    parsed: Dict[str, Any],
    expected_action: str,
    expected_target_node: str,
    tool_calls: int,
) -> float:
    reward = 0.08
    if parsed.get("action") == expected_action:
        reward += 0.62
    if parsed.get("target_node") == expected_target_node:
        reward += 0.18
    if parsed.get("action") == expected_action and tool_calls >= 2:
        reward += 0.07
    if parsed.get("action") == "monitor" and expected_action != "monitor":
        reward -= 0.08
    return max(0.01, min(0.99, round(reward, 2)))


def build_grpo_reward_fn():
    def reward_fn(completions: List[Any], **kwargs: Any) -> List[float]:
        phases = kwargs.get("phase", [])
        expected_threat_types = kwargs.get("expected_threat_type", [])
        expected_targets = kwargs.get("expected_target_node", [])
        expected_recommended_actions = kwargs.get("expected_recommended_action", [])
        expected_actions = kwargs.get("expected_action", [])
        tool_calls = kwargs.get("tool_calls", [])
        rewards: List[float] = []
        for index, completion in enumerate(completions):
            phase = int(phases[index]) if phases else 1
            text = _completion_to_text(completion)
            parsed = parse_response(text, phase)
            if phase == 1:
                reward = _phase1_reward(
                    parsed=parsed,
                    expected_threat_type=str(expected_threat_types[index]),
                    expected_target_node=str(expected_targets[index]),
                    expected_recommended_action=str(expected_recommended_actions[index]),
                )
            else:
                reward = _phase2_reward(
                    parsed=parsed,
                    expected_action=str(expected_actions[index]),
                    expected_target_node=str(expected_targets[index]),
                    tool_calls=int(tool_calls[index]) if tool_calls else 0,
                )
            rewards.append(reward)
        return rewards

    return reward_fn


def _filter_supported_kwargs(callable_obj: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return kwargs
    valid = {}
    for key, value in kwargs.items():
        if key in signature.parameters:
            valid[key] = value
    return valid


def _trainer_log_rows(log_history: List[Dict[str, Any]], selected_task: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for entry in log_history:
        step = entry.get("step")
        if step is None:
            continue
        reward_keys = [
            "reward",
            "mean_reward",
            "rewards/mean",
            "objective",
            "objective/rlhf_reward",
        ]
        score = None
        for key in reward_keys:
            if key in entry:
                try:
                    score = float(entry[key])
                    break
                except Exception:
                    continue
        if score is None:
            score = 0.50
        row = {
            "episode": int(step),
            "task": "mixed" if selected_task == "all" else selected_task,
            "stage": "grpo",
            "score": max(0.01, min(0.99, score)),
            "loss": float(entry.get("loss", 0.0) or 0.0),
            "learning_rate": float(entry.get("learning_rate", 0.0) or 0.0),
        }
        rows.append(row)
    return rows


def evaluate_model_suite(
    model: Any,
    tokenizer: Any,
    selected_task: str,
    eval_episodes: int,
    max_steps: int,
    use_tools: bool,
) -> List[Dict[str, Any]]:
    tasks = TASKS if selected_task == "all" else [selected_task]
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        scores: List[float] = []
        steps: List[int] = []
        tool_calls: List[int] = []
        for _ in range(eval_episodes):
            _, metrics = run_model_episode(
                model=model,
                tokenizer=tokenizer,
                task=task,
                max_steps=max_steps,
                use_tools=use_tools,
            )
            scores.append(float(metrics["score"]))
            steps.append(int(metrics["steps"]))
            tool_calls.append(int(metrics["tool_calls"]))
        rows.append({
            "episode": len(rows) + 1,
            "task": task,
            "stage": "evaluation",
            "score": round(sum(scores) / len(scores), 3) if scores else 0.50,
            "steps": round(sum(steps) / len(steps), 2) if steps else 0.0,
            "tool_calls": round(sum(tool_calls) / len(tool_calls), 2) if tool_calls else 0.0,
            "eval_episodes": eval_episodes,
        })
    return rows


def run_model_episode(
    model: Any,
    tokenizer: Any,
    task: str,
    max_steps: int,
    use_tools: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    env = AdaptShieldEnvironment(task_name=task)
    obs = env.reset()
    samples: List[Dict[str, Any]] = []
    rewards: List[float] = []
    tool_calls = 0

    while not obs.done and len(samples) < max_steps:
        phase = int(getattr(obs, "phase", 1))
        tool_results = investigate_local_with_depth(
            env,
            obs,
            use_tools=use_tools,
            thorough=True,
        )
        tool_calls += len(tool_results)
        obs_dict = obs_to_dict(obs)
        obs_dict = attach_tool_results(obs_dict, tool_results)
        messages = build_messages(obs_dict)
        prompt, response = generate_response(model, tokenizer, messages)
        payload = parse_response(response, phase)

        try:
            obs = env.step(AdaptShieldAction(**payload))
            reward = float(obs.reward)
        except Exception as exc:
            reward = 0.01
            samples.append({
                "prompt": prompt,
                "response": response,
                "reward": reward,
                "phase": phase,
                "tool_calls": len(tool_results),
                "error": str(exc),
            })
            break

        rewards.append(reward)
        samples.append({
            "prompt": prompt,
            "response": response,
            "reward": reward,
            "phase": phase,
            "tool_calls": len(tool_results),
            "error": None,
        })

    metadata = obs.metadata if isinstance(obs.metadata, dict) else {}
    if "normalized_score" not in metadata:
        raise RuntimeError("normalized_score missing after training episode")

    return samples, {
        "score": float(metadata["normalized_score"]),
        "steps": len(samples),
        "reward_sum": sum(rewards),
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
        "tool_calls": tool_calls,
    }


def train_policy_gradient(args: argparse.Namespace) -> None:
    from unsloth import FastLanguageModel
    import torch
    from torch.optim import AdamW

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_name = MODEL_CHOICES[args.model]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("AdaptShield policy-gradient GPU training")
    print(f"Task: {args.task}")
    print(f"Curriculum: {args.curriculum}")
    print(f"Use tools: {args.use_tools}")
    print(f"Model: {model_name}")
    print(f"Episodes: {args.episodes}")
    print(f"Output: {output_dir}")
    print()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    FastLanguageModel.for_training(model)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    rows: List[Dict[str, Any]] = []
    best_score = -1.0
    for episode in range(1, args.episodes + 1):
        started = time.time()
        task, stage = task_for_episode(
            episode=episode,
            total_episodes=args.episodes,
            selected_task=args.task,
            curriculum=args.curriculum,
        )
        samples, metrics = run_model_episode(
            model=model,
            tokenizer=tokenizer,
            task=task,
            max_steps=args.max_steps,
            use_tools=args.use_tools,
        )
        rewards = [float(sample["reward"]) for sample in samples]
        baseline = sum(rewards) / len(rewards) if rewards else 0.0
        total_loss = 0.0

        for sample in samples:
            advantage = float(sample["reward"]) - baseline
            full_text = sample["prompt"] + sample["response"] + tokenizer.eos_token
            inputs = tokenizer(
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=MAX_SEQ_LEN,
            ).to("cuda")
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss * (-advantage)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())

        row = {
            "episode": episode,
            "task": task,
            "stage": stage,
            "score": metrics["score"],
            "steps": metrics["steps"],
            "reward_sum": metrics["reward_sum"],
            "mean_reward": metrics["mean_reward"],
            "tool_calls": metrics["tool_calls"],
            "loss": total_loss,
            "seconds": round(time.time() - started, 2),
        }
        rows.append(row)

        print(
            f"episode={episode:03d} task={task:<20} "
            f"stage={stage:<32} "
            f"score={row['score']:.3f} mean_reward={row['mean_reward']:.3f} "
            f"loss={row['loss']:.4f} steps={row['steps']:02d} tools={row['tool_calls']:02d}"
        )

        if row["score"] > best_score:
            best_score = row["score"]
            model.save_pretrained(output_dir / "best")
            tokenizer.save_pretrained(output_dir / "best")

        if args.save_every and episode % args.save_every == 0:
            model.save_pretrained(output_dir / f"checkpoint-{episode}")
            tokenizer.save_pretrained(output_dir / f"checkpoint-{episode}")

    model.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")

    evaluation_rows = evaluate_model_suite(
        model=model,
        tokenizer=tokenizer,
        selected_task=args.task,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        use_tools=args.use_tools,
    )

    metrics_path = save_metrics(
        output_dir=output_dir,
        rows=rows,
        model_name=model_name,
        episodes=args.episodes,
        curriculum=args.curriculum,
        use_tools=args.use_tools,
        trainer="pg",
        evaluation_rows=evaluation_rows,
    )
    if args.plot:
        maybe_plot(metrics_path, output_dir)
    print()
    print(f"Training complete. Best score: {best_score:.3f}")
    print("Post-train online evaluation:")
    for row in evaluation_rows:
        print(
            f"  task={row['task']:<20} score={row['score']:.3f} "
            f"steps={row['steps']} tools={row['tool_calls']}"
        )
    print(f"Metrics saved to: {metrics_path}")


def train_grpo(args: argparse.Namespace) -> None:
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer
    from unsloth import FastLanguageModel
    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_name = MODEL_CHOICES[args.model]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("AdaptShield GRPO training")
    print(f"Task: {args.task}")
    print(f"Curriculum: {args.curriculum}")
    print(f"Use tools: {args.use_tools}")
    print(f"Model: {model_name}")
    print(f"Prompt-bank episodes: {args.prompt_bank_episodes}")
    print(f"GRPO epochs: {args.grpo_epochs}")
    print(f"Eval episodes: {args.eval_episodes}")
    print(f"Output: {output_dir}")
    print()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=LORA_RANK * 2,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token
    FastLanguageModel.for_training(model)

    prompt_bank = build_prompt_bank(
        tokenizer=tokenizer,
        selected_task=args.task,
        curriculum=args.curriculum,
        rollout_episodes=args.prompt_bank_episodes,
        max_steps=args.max_steps,
        use_tools=args.use_tools,
        seed=args.seed,
    )
    if not prompt_bank:
        raise RuntimeError("Prompt bank is empty; cannot start GRPO training.")

    dataset = Dataset.from_list(prompt_bank)
    reward_fn = build_grpo_reward_fn()

    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())
    config_kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.grpo_epochs,
        "max_prompt_length": MAX_SEQ_LEN - 256,
        "max_completion_length": 256,
        "num_generations": args.num_generations,
        "logging_steps": 1,
        "save_steps": args.save_every,
        "save_strategy": "steps",
        "report_to": "none",
        "remove_unused_columns": False,
        "bf16": bf16_supported,
        "fp16": not bf16_supported,
        "seed": args.seed,
    }
    grpo_config = GRPOConfig(**_filter_supported_kwargs(GRPOConfig, config_kwargs))

    trainer_kwargs = {
        "model": model,
        "reward_funcs": [reward_fn],
        "args": grpo_config,
        "train_dataset": dataset,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
    }
    trainer = GRPOTrainer(**_filter_supported_kwargs(GRPOTrainer, trainer_kwargs))
    trainer.train()

    model.save_pretrained(output_dir / "final")
    tokenizer.save_pretrained(output_dir / "final")

    log_history = list(getattr(getattr(trainer, "state", None), "log_history", []) or [])
    train_rows = _trainer_log_rows(log_history, selected_task=args.task)
    if not train_rows:
        train_rows = [{
            "episode": index + 1,
            "task": "mixed" if args.task == "all" else args.task,
            "stage": "grpo",
            "score": 0.50,
        } for index in range(max(1, args.grpo_epochs))]

    evaluation_rows = evaluate_model_suite(
        model=model,
        tokenizer=tokenizer,
        selected_task=args.task,
        eval_episodes=args.eval_episodes,
        max_steps=args.max_steps,
        use_tools=args.use_tools,
    )

    metrics_path = save_metrics(
        output_dir=output_dir,
        rows=train_rows,
        model_name=model_name,
        episodes=max(1, len(train_rows)),
        curriculum=args.curriculum,
        use_tools=args.use_tools,
        trainer="grpo",
        evaluation_rows=evaluation_rows,
        prompt_bank_size=len(prompt_bank),
    )
    if args.plot:
        maybe_plot(metrics_path, output_dir)
    print("GRPO training complete.")
    print(f"Prompt bank size: {len(prompt_bank)}")
    print("Post-train online evaluation:")
    for row in evaluation_rows:
        print(
            f"  task={row['task']:<20} score={row['score']:.3f} "
            f"steps={row['steps']} tools={row['tool_calls']}"
        )
    if log_history:
        final_keys = sorted(log_history[-1].keys())
        print(f"Trainer log keys: {final_keys}")
    print(f"Metrics saved to: {metrics_path}")


def run_fallback_smoke(args: argparse.Namespace) -> None:
    if args.use_tools:
        run_tool_fallback_smoke(args)
        return

    if args.curriculum:
        tasks = [
            task_for_episode(
                episode=episode,
                total_episodes=min(args.episodes, args.smoke_episodes),
                selected_task=args.task,
                curriculum=True,
            )[0]
            for episode in range(1, min(args.episodes, args.smoke_episodes) + 1)
        ]
    else:
        tasks = TASKS if args.task == "all" else [args.task]

    rows = run_smoke_training(
        tasks=tasks,
        episodes=min(args.episodes, args.smoke_episodes),
        output=Path(args.output) / "train_smoke.csv",
        seed=args.seed,
        epsilon=0.85,
        epsilon_decay=0.94,
        epsilon_floor=0.08,
        lr=0.35,
        max_steps=args.max_steps,
    )
    output_dir = Path(args.output)
    metrics_rows = []
    for row in rows:
        row = dict(row)
        episode = int(row["episode"])
        _, stage = task_for_episode(
            episode=episode,
            total_episodes=min(args.episodes, args.smoke_episodes),
            selected_task=args.task,
            curriculum=args.curriculum,
        )
        row["stage"] = stage
        metrics_rows.append(row)

    metrics_path = save_metrics(
        output_dir=output_dir,
        rows=metrics_rows,
        model_name="smoke-tabular-policy",
        episodes=min(args.episodes, args.smoke_episodes),
        curriculum=args.curriculum,
        use_tools=False,
    )
    print(f"Metrics saved to: {metrics_path}")
    if args.plot:
        maybe_plot(metrics_path, output_dir)


def run_tool_fallback_smoke(args: argparse.Namespace) -> None:
    """No-GPU tool-aware rehearsal. This validates flow, not model learning."""
    from tool_baseline import run_task as run_tool_task

    total = min(args.episodes, args.smoke_episodes)
    if args.curriculum:
        tasks = [
            task_for_episode(
                episode=episode,
                total_episodes=total,
                selected_task=args.task,
                curriculum=True,
            )[0]
            for episode in range(1, total + 1)
        ]
    else:
        tasks = TASKS if args.task == "all" else [args.task]

    print("AdaptShield tool-aware smoke evaluation")
    print("Mode: no-GPU flow validation, not model learning")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Episodes: {total}")
    print()

    rows: List[Dict[str, Any]] = []
    for episode in range(1, total + 1):
        task = tasks[(episode - 1) % len(tasks)]
        result = run_tool_task(task, emit_logs=False)
        metadata = result.get("metadata", {})
        tool_calls = len(metadata.get("tool_trace", [])) if isinstance(metadata, dict) else 0
        _, stage = task_for_episode(
            episode=episode,
            total_episodes=total,
            selected_task=args.task,
            curriculum=args.curriculum,
        )
        row = {
            "episode": episode,
            "task": task,
            "stage": stage,
            "score": result["score"],
            "steps": result["steps"],
            "reward_sum": sum(result["rewards"]),
            "mean_reward": sum(result["rewards"]) / len(result["rewards"]) if result["rewards"] else 0.0,
            "tool_calls": tool_calls,
            "status": "PASS" if result["success"] else "FAIL",
        }
        rows.append(row)
        print(
            f"episode={episode:03d} task={task:<20} "
            f"score={row['score']:.3f} steps={row['steps']:02d} "
            f"tools={tool_calls:02d} {row['status']}"
        )

    output_dir = Path(args.output)
    metrics_path = save_metrics(
        output_dir=output_dir,
        rows=rows,
        model_name="tool-aware-smoke-policy",
        episodes=total,
        curriculum=args.curriculum,
        use_tools=True,
    )
    print(f"Metrics saved to: {metrics_path}")
    if args.plot:
        maybe_plot(metrics_path, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AdaptShield training harness.")
    parser.add_argument("--task", default="direct-triage", choices=TASKS + ["all"])
    parser.add_argument("--model", default=DEFAULT_MODEL, choices=list(MODEL_CHOICES))
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--output", default="checkpoints/adaptshield")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--save-every", type=int, default=20)
    parser.add_argument("--smoke", action="store_true", help="Force dependency-free smoke mode.")
    parser.add_argument("--smoke-episodes", type=int, default=30)
    parser.add_argument("--curriculum", action="store_true", help="Train direct -> dual -> hard instead of fixed/round-robin tasks.")
    parser.add_argument("--use-tools", action="store_true", help="Let GPU training query SOC tools before hard-task actions.")
    parser.add_argument("--plot", action="store_true", help="Generate reward_curve.png from metrics.json after training.")
    parser.add_argument("--trainer", default="auto", choices=["auto", "pg", "grpo"], help="Training backend: safe policy-gradient fallback or TRL GRPO.")
    parser.add_argument("--prompt-bank-episodes", type=int, default=24, help="Reference rollout episodes used to build the GRPO prompt bank.")
    parser.add_argument("--grpo-epochs", type=int, default=1, help="Number of epochs over the prompt bank for GRPO runs.")
    parser.add_argument("--num-generations", type=int, default=4, help="GRPO generations per prompt when TRL path is active.")
    parser.add_argument("--per-device-batch-size", type=int, default=1, help="Per-device batch size for GRPO training.")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4, help="Gradient accumulation for GRPO training.")
    parser.add_argument("--eval-episodes", type=int, default=2, help="Online environment episodes per task after GPU training.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.smoke:
        run_fallback_smoke(args)
        return 0

    trainer_choice = args.trainer
    if trainer_choice == "auto":
        try:
            import datasets  # noqa: F401
            import trl  # noqa: F401
            trainer_choice = "grpo"
        except ImportError:
            trainer_choice = "pg"

    try:
        if trainer_choice == "grpo":
            train_grpo(args)
        else:
            train_policy_gradient(args)
    except ImportError as exc:
        print(f"GPU training dependency missing for trainer={trainer_choice}: {exc}")
        if trainer_choice == "grpo":
            print("Falling back to policy-gradient GPU trainer.")
            try:
                train_policy_gradient(args)
                return 0
            except ImportError as nested_exc:
                print(f"Policy-gradient fallback also unavailable: {nested_exc}")
        print("Falling back to dependency-free smoke training.")
        run_fallback_smoke(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
