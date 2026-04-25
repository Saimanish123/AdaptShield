#!/usr/bin/env python3
"""Supervised fine-tuning for AdaptShield chat-style demonstrations."""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List

from train import (
    DEFAULT_MODEL,
    LORA_RANK,
    MAX_SEQ_LEN,
    MODEL_CHOICES,
    _align_trainable_dtypes,
    _filter_supported_kwargs,
    _normalize_generation_config,
    evaluate_model_suite,
    run_model_episode,
)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"No training rows found in {path}")
    return rows


def build_loss_plot(log_history: List[Dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping loss plot")
        return

    xs: List[int] = []
    ys: List[float] = []
    for index, entry in enumerate(log_history, start=1):
        if "loss" not in entry:
            continue
        step = int(entry.get("step", index) or index)
        try:
            loss = float(entry["loss"])
        except Exception:
            continue
        xs.append(step)
        ys.append(loss)

    if not xs:
        print("No loss entries found; skipping loss plot")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(xs, ys, color="#0f4c81", linewidth=2, label="training loss")
    plt.xlabel("Training step")
    plt.ylabel("Loss")
    plt.title("AdaptShield SFT Loss Curve")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def build_reward_plot(rows: List[Dict[str, Any]], output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping reward plot")
        return

    if not rows:
        print("No held-out reward rows found; skipping reward plot")
        return

    checkpoint_labels = [str(row["checkpoint"]) for row in rows]
    in_distribution_scores = [float(row["in_distribution_score"]) for row in rows]
    heldout_scores = [float(row["heldout_score"]) for row in rows]

    plt.figure(figsize=(10, 5))
    plt.plot(
        range(len(rows)),
        in_distribution_scores,
        color="#136f63",
        linewidth=2.5,
        marker="o",
        label="in-distribution mean reward",
    )
    plt.plot(
        range(len(rows)),
        heldout_scores,
        color="#8a3ffc",
        linewidth=2.5,
        marker="s",
        label="held-out family mean reward",
    )
    plt.xticks(range(len(rows)), checkpoint_labels, rotation=35, ha="right")
    plt.xlabel("Checkpoint")
    plt.ylabel("normalized_score")
    plt.title("AdaptShield In-Distribution vs Held-out Reward Curve")
    plt.ylim(0.0, 1.0)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def render_example(example: Dict[str, Any], tokenizer: Any) -> str:
    if "messages" in example:
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    return str(example["text"])


def _checkpoint_sort_key(path: Path) -> tuple[int, str]:
    if path.name == "final":
        return (10**9, path.name)
    if path.name.startswith("checkpoint-"):
        try:
            return (int(path.name.split("-", 1)[1]), path.name)
        except Exception:
            return (10**8, path.name)
    return (10**7, path.name)


def checkpoint_dirs(output_dir: Path) -> List[Path]:
    checkpoints = [
        path for path in output_dir.iterdir()
        if path.is_dir() and (path.name.startswith("checkpoint-") or path.name == "final")
    ]
    return sorted(checkpoints, key=_checkpoint_sort_key)


def evaluate_suite_with_seed(
    model: Any,
    tokenizer: Any,
    selected_task: str,
    eval_episodes: int,
    max_steps: int,
    use_tools: bool,
    seed_start: int,
    world_split: str,
    world_family: str | None = None,
) -> List[Dict[str, Any]]:
    tasks = ["direct-triage", "dual-pivot", "polymorphic-zero-day"] if selected_task == "all" else [selected_task]
    rows: List[Dict[str, Any]] = []
    original_seed = os.environ.get("ADAPTSHIELD_SEED")
    try:
        for task_index, task in enumerate(tasks):
            scores: List[float] = []
            steps: List[int] = []
            tool_calls: List[int] = []
            for episode_index in range(eval_episodes):
                os.environ["ADAPTSHIELD_SEED"] = str(seed_start + task_index * 100 + episode_index)
                _, metrics = run_model_episode(
                model=model,
                tokenizer=tokenizer,
                task=task,
                max_steps=max_steps,
                use_tools=use_tools,
                world_split=world_split,
                world_family=world_family,
            )
                scores.append(float(metrics["score"]))
                steps.append(int(metrics["steps"]))
                tool_calls.append(int(metrics["tool_calls"]))
            rows.append({
                "task": task,
                "score": round(sum(scores) / len(scores), 3) if scores else 0.50,
                "steps": round(sum(steps) / len(steps), 2) if steps else 0.0,
                "tool_calls": round(sum(tool_calls) / len(tool_calls), 2) if tool_calls else 0.0,
                "eval_episodes": eval_episodes,
                "seed_start": seed_start,
                "world_split": world_split,
                "world_family": world_family or "auto",
            })
    finally:
        if original_seed is None:
            os.environ.pop("ADAPTSHIELD_SEED", None)
        else:
            os.environ["ADAPTSHIELD_SEED"] = original_seed
    return rows


def evaluate_saved_checkpoints(
    output_dir: Path,
    model_key: str,
    max_seq_length: int,
    selected_task: str,
    eval_episodes: int,
    max_steps: int,
    use_tools: bool,
    heldout_seed: int,
    train_world_split: str,
    heldout_world_split: str,
) -> List[Dict[str, Any]]:
    from unsloth import FastLanguageModel

    rows: List[Dict[str, Any]] = []
    for index, checkpoint_dir in enumerate(checkpoint_dirs(output_dir)):
        print(f"Held-out evaluating checkpoint: {checkpoint_dir.name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=str(checkpoint_dir),
            max_seq_length=max_seq_length,
            load_in_4bit=True,
            dtype=None,
        )
        _normalize_generation_config(model)
        _align_trainable_dtypes(model)
        in_distribution_rows = evaluate_suite_with_seed(
            model=model,
            tokenizer=tokenizer,
            selected_task=selected_task,
            eval_episodes=eval_episodes,
            max_steps=max_steps,
            use_tools=use_tools,
            seed_start=heldout_seed + index * 1000,
            world_split=train_world_split,
        )
        heldout_rows = evaluate_suite_with_seed(
            model=model,
            tokenizer=tokenizer,
            selected_task=selected_task,
            eval_episodes=eval_episodes,
            max_steps=max_steps,
            use_tools=use_tools,
            seed_start=heldout_seed + index * 1000,
            world_split=heldout_world_split,
        )
        in_distribution_score = round(
            sum(float(row["score"]) for row in in_distribution_rows) / max(1, len(in_distribution_rows)),
            3,
        )
        heldout_score = round(
            sum(float(row["score"]) for row in heldout_rows) / max(1, len(heldout_rows)),
            3,
        )
        rows.append({
            "checkpoint": checkpoint_dir.name,
            "in_distribution_score": in_distribution_score,
            "heldout_score": heldout_score,
            "in_distribution_rows": in_distribution_rows,
            "heldout_rows": heldout_rows,
        })
        del model
        del tokenizer
    return rows


def train_sft(args: argparse.Namespace) -> None:
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer
    import torch

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_path = Path(args.dataset)
    rows = load_jsonl(dataset_path)
    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]

    model_name = MODEL_CHOICES[args.model]
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("AdaptShield SFT training")
    print(f"Dataset: {dataset_path}")
    print(f"Rows: {len(rows)}")
    print(f"Model: {model_name}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.per_device_batch_size}")
    print(f"Grad accumulation: {args.gradient_accumulation_steps}")
    print(f"Learning rate: {args.lr}")
    print(f"Output: {output_dir}")
    print()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=args.max_seq_length,
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
    _normalize_generation_config(model)
    _align_trainable_dtypes(model)

    prepared_rows = [{"text": render_example(row, tokenizer), **row} for row in rows]
    dataset = Dataset.from_list(prepared_rows)

    bf16_supported = bool(getattr(torch.cuda, "is_bf16_supported", lambda: False)())

    try:
        from trl import SFTConfig
        train_config_cls = SFTConfig
    except ImportError:
        from transformers import TrainingArguments
        train_config_cls = TrainingArguments

    config_kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.per_device_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.epochs,
        "logging_steps": 1,
        "save_strategy": "steps",
        "save_steps": args.save_steps,
        "report_to": "none",
        "seed": args.seed,
        "bf16": bf16_supported,
        "fp16": not bf16_supported,
        "max_seq_length": args.max_seq_length,
        "dataset_text_field": "text",
        "dataset_num_proc": 1,
        "packing": False,
    }
    train_args = train_config_cls(
        **_filter_supported_kwargs(train_config_cls, config_kwargs)
    )

    trainer_kwargs = {
        "model": model,
        "train_dataset": dataset,
        "args": train_args,
        "processing_class": tokenizer,
        "tokenizer": tokenizer,
        "dataset_text_field": "text",
        "dataset_num_proc": 1,
        "max_seq_length": args.max_seq_length,
        "packing": False,
    }
    trainer = SFTTrainer(**_filter_supported_kwargs(SFTTrainer, trainer_kwargs))
    trainer.train()

    final_dir = output_dir / "final"
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    log_history = list(getattr(getattr(trainer, "state", None), "log_history", []) or [])
    loss_plot_path = output_dir / "loss_curve.png"
    build_loss_plot(log_history, loss_plot_path)

    evaluation_rows = evaluate_suite_with_seed(
        model=model,
        tokenizer=tokenizer,
        selected_task=args.eval_task,
        eval_episodes=args.eval_episodes,
        max_steps=args.eval_max_steps,
        use_tools=args.use_tools,
        seed_start=args.heldout_seed,
        world_split=args.train_world_split,
    )
    heldout_evaluation_rows = evaluate_suite_with_seed(
        model=model,
        tokenizer=tokenizer,
        selected_task=args.eval_task,
        eval_episodes=args.eval_episodes,
        max_steps=args.eval_max_steps,
        use_tools=args.use_tools,
        seed_start=args.heldout_seed,
        world_split=args.heldout_world_split,
    )

    reward_curve_rows = evaluate_saved_checkpoints(
        output_dir=output_dir,
        model_key=args.model,
        max_seq_length=args.max_seq_length,
        selected_task=args.eval_task,
        eval_episodes=args.eval_episodes,
        max_steps=args.eval_max_steps,
        use_tools=args.use_tools,
        heldout_seed=args.heldout_seed,
        train_world_split=args.train_world_split,
        heldout_world_split=args.heldout_world_split,
    )
    reward_plot_path = output_dir / "reward_curve.png"
    build_reward_plot(reward_curve_rows, reward_plot_path)

    metrics = {
        "trainer": "sft",
        "model": model_name,
        "dataset": str(dataset_path),
        "rows": len(rows),
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "evaluation_rows": evaluation_rows,
        "heldout_evaluation_rows": heldout_evaluation_rows,
        "heldout_seed": args.heldout_seed,
        "train_world_split": args.train_world_split,
        "heldout_world_split": args.heldout_world_split,
        "reward_curve_rows": reward_curve_rows,
        "log_history": log_history,
    }
    metrics_path = output_dir / "sft_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("SFT complete.")
    print(f"Saved adapter to: {final_dir}")
    print(f"Loss curve: {loss_plot_path}")
    print(f"Reward curve: {reward_plot_path}")
    print(f"Metrics: {metrics_path}")
    print("Post-train evaluation:")
    for row in evaluation_rows:
        print(
            f"  task={row['task']:<20} score={row['score']:.3f} "
            f"steps={row['steps']} tools={row['tool_calls']}"
        )
    print("Held-out checkpoint reward curve:")
    for row in reward_curve_rows:
        print(
            f"  checkpoint={row['checkpoint']:<16} "
            f"in_dist={row['in_distribution_score']:.3f} "
            f"heldout={row['heldout_score']:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="AdaptShield supervised fine-tuning")
    parser.add_argument(
        "--dataset",
        default="data/adaptshield_sft.jsonl",
        help="Path to JSONL dataset from generate_sft_data.py",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=list(MODEL_CHOICES.keys()),
    )
    parser.add_argument("--output", default="checkpoints/sft-run")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heldout-seed", type=int, default=314)
    parser.add_argument("--train-world-split", default="train", choices=["train", "eval"])
    parser.add_argument("--heldout-world-split", default="eval", choices=["train", "eval"])
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=40)
    parser.add_argument(
        "--eval-task",
        default="all",
        choices=["all", "direct-triage", "dual-pivot", "polymorphic-zero-day"],
    )
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--eval-max-steps", type=int, default=20)
    parser.add_argument(
        "--use-tools",
        action="store_true",
        help="Use SOC tools during post-train evaluation.",
    )
    args = parser.parse_args()
    train_sft(args)


if __name__ == "__main__":
    main()
