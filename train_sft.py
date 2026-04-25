#!/usr/bin/env python3
"""Supervised fine-tuning for AdaptShield chat-style demonstrations."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from train import (
    DEFAULT_MODEL,
    LORA_RANK,
    MAX_SEQ_LEN,
    MODEL_CHOICES,
    _filter_supported_kwargs,
    evaluate_model_suite,
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


def render_example(example: Dict[str, Any], tokenizer: Any) -> str:
    if "messages" in example:
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
    return str(example["text"])


def train_sft(args: argparse.Namespace) -> None:
    from datasets import Dataset
    from trl import SFTTrainer
    from unsloth import FastLanguageModel
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
        "save_strategy": "epoch",
        "report_to": "none",
        "seed": args.seed,
        "bf16": bf16_supported,
        "fp16": not bf16_supported,
        "max_seq_length": args.max_seq_length,
        "dataset_text_field": "text",
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

    evaluation_rows = evaluate_model_suite(
        model=model,
        tokenizer=tokenizer,
        selected_task=args.eval_task,
        eval_episodes=args.eval_episodes,
        max_steps=args.eval_max_steps,
        use_tools=args.use_tools,
    )

    metrics = {
        "trainer": "sft",
        "model": model_name,
        "dataset": str(dataset_path),
        "rows": len(rows),
        "epochs": args.epochs,
        "learning_rate": args.lr,
        "evaluation_rows": evaluation_rows,
        "log_history": log_history,
    }
    metrics_path = output_dir / "sft_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("SFT complete.")
    print(f"Saved adapter to: {final_dir}")
    print(f"Loss curve: {loss_plot_path}")
    print(f"Metrics: {metrics_path}")
    print("Post-train evaluation:")
    for row in evaluation_rows:
        print(
            f"  task={row['task']:<20} score={row['score']:.3f} "
            f"steps={row['steps']} tools={row['tool_calls']}"
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
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LEN)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
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
