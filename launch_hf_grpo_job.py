#!/usr/bin/env python3
"""Launch AdaptShield GRPO refinement on Hugging Face Jobs."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import time
from pathlib import Path

from huggingface_hub import HfApi, get_token, run_job
from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

from train import MODEL_CHOICES


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_IMAGE = "pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel"


def _should_retry_hf(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    return status_code == 429 or (isinstance(status_code, int) and 500 <= status_code < 600)


def _retry_hf_call(fn, *args, retries: int = 4, delay_s: float = 2.0, **kwargs):
    last_exc = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if not _should_retry_hf(exc) or attempt == retries - 1:
                raise
            sleep_for = delay_s * (2 ** attempt)
            print(f"Retrying HF API call after transient error ({exc}); sleeping {sleep_for:.1f}s")
            time.sleep(sleep_for)
    raise last_exc  # pragma: no cover


def infer_repo_url() -> str:
    result = subprocess.run(
        ["git", "config", "--get", "remote.origin.url"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    repo_url = result.stdout.strip()
    if not repo_url:
        raise RuntimeError("Could not infer git remote.origin.url")
    return repo_url


def repo_namespace(repo_id: str) -> str:
    if "/" not in repo_id:
        raise RuntimeError(f"Invalid repo id: {repo_id}. Expected namespace/name.")
    return repo_id.split("/", 1)[0]


def authenticated_username(api: HfApi) -> str | None:
    try:
        info = api.whoami(cache=True)
    except Exception:
        return None
    if isinstance(info, dict):
        for key in ("name", "fullname", "user"):
            value = info.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def validate_repo_access(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    skip_create: bool,
    allow_cross_namespace: bool,
) -> None:
    owner = repo_namespace(repo_id)
    username = authenticated_username(api)
    if username and owner != username:
        message = (
            f"Authenticated HF account appears to be '{username}', but target repo is under '{owner}'. "
            "Use a repo under the same namespace or pass --allow-cross-namespace only if you are certain "
            "this token has write access there."
        )
        if not allow_cross_namespace:
            raise RuntimeError(message)
        print(f"Warning: {message}")

    if skip_create or repo_type == "model":
        try:
            _retry_hf_call(api.repo_info, repo_id=repo_id, repo_type=repo_type)
        except RepositoryNotFoundError as exc:
            raise RuntimeError(
                f"Repo '{repo_id}' ({repo_type}) was not found or is not accessible with the current token."
            ) from exc
        except HfHubHTTPError as exc:
            raise RuntimeError(f"Could not verify repo '{repo_id}' ({repo_type}): {exc}") from exc


def validate_source_artifacts(
    api: HfApi,
    repo_id: str,
    repo_type: str,
    subdir: str,
) -> None:
    try:
        files = set(_retry_hf_call(api.list_repo_files, repo_id=repo_id, repo_type=repo_type))
    except Exception as exc:
        raise RuntimeError(f"Could not list files for source repo '{repo_id}' ({repo_type}): {exc}") from exc

    required = {
        f"{subdir}/final/adapter_config.json",
        f"{subdir}/sft_metrics.json",
    }
    missing = sorted(path for path in required if path not in files)
    if missing:
        raise RuntimeError(
            "Source repo is missing required SFT artifacts: " + ", ".join(missing)
        )


def build_command(args: argparse.Namespace, repo_url: str, output_subdir: str) -> str:
    output_path = f"/workspace/adaptshield/checkpoints/{output_subdir}"

    return f"""
set -euo pipefail
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONWARNINGS="ignore::FutureWarning"
apt-get update
apt-get install -y git
git clone {shlex.quote(repo_url)} /workspace/adaptshield
cd /workspace/adaptshield
python -m pip install -U pip setuptools wheel
pip install -e .
pip uninstall -y torchaudio || true
pip install matplotlib unsloth trl accelerate bitsandbytes huggingface_hub mergekit
python - <<'PY'
import importlib
for name in ["torch", "transformers", "trl", "unsloth", "peft", "mergekit", "train", "build_benchmark_table"]:
    importlib.import_module(name)
print("Dependency smoke check passed.")
PY

python - <<'PY'
from huggingface_hub import snapshot_download
from pathlib import Path

repo_id = {args.source_repo!r}
repo_type = {args.source_repo_type!r}
subdir = {args.source_subdir!r}
local_dir = snapshot_download(repo_id=repo_id, repo_type=repo_type)
adapter_path = Path(local_dir) / subdir / "final"
sft_metrics_path = Path(local_dir) / subdir / "sft_metrics.json"
if not adapter_path.exists():
    raise RuntimeError(f"SFT adapter path not found: {{adapter_path}}")
if not sft_metrics_path.exists():
    raise RuntimeError(f"SFT metrics path not found: {{sft_metrics_path}}")
print(adapter_path)
Path("/workspace/adaptshield/.grpo_adapter_path.txt").write_text(str(adapter_path), encoding="utf-8")
Path("/workspace/adaptshield/.grpo_sft_metrics_path.txt").write_text(str(sft_metrics_path), encoding="utf-8")
PY

ADAPTER_PATH=$(cat /workspace/adaptshield/.grpo_adapter_path.txt)
SFT_METRICS_PATH=$(cat /workspace/adaptshield/.grpo_sft_metrics_path.txt)

python train.py \\
  --trainer grpo \\
  --task all \\
  --curriculum \\
  --use-tools \\
  --model {args.model} \\
  --model-path "$ADAPTER_PATH" \\
  --prompt-bank-episodes {args.prompt_bank_episodes} \\
  --max-steps {args.max_steps} \\
  --prompt-bank-hard-multiplier {args.prompt_bank_hard_multiplier} \\
  --prompt-bank-borderline-bonus {args.prompt_bank_borderline_bonus} \\
  --grpo-epochs {args.grpo_epochs} \\
  --num-generations {args.num_generations} \\
  --per-device-batch-size {args.per_device_batch_size} \\
  --gradient-accumulation-steps {args.gradient_accumulation_steps} \\
  --save-every {args.save_every} \\
  --eval-episodes {args.eval_episodes} \\
  --train-world-split train \\
  --heldout-world-split eval \\
  --heldout-seed {args.heldout_seed} \\
  --output {output_path} \\
  --plot

if ! python build_benchmark_table.py \\
  --sft-metrics "$SFT_METRICS_PATH" \\
  --grpo-metrics {output_path}/metrics.json \\
  --output {output_path}/benchmark_table.md; then
  echo "Benchmark table generation failed; continuing with core artifacts."
fi

python - <<'PY'
import os
import time
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = os.environ["RUNS_REPO"]
repo_type = os.environ["RUNS_REPO_TYPE"]
output_dir = {output_path!r}
subdir = {output_subdir!r}

last_exc = None
for attempt in range(4):
    try:
        api.upload_folder(
            repo_id=repo_id,
            repo_type=repo_type,
            folder_path=output_dir,
            path_in_repo=subdir,
        )
        last_exc = None
        break
    except Exception as exc:
        last_exc = exc
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)
        if status_code == 429 or (isinstance(status_code, int) and 500 <= status_code < 600):
            sleep_for = 2 ** attempt
            print(f"Transient upload error: {{exc}}; retrying in {{sleep_for}}s")
            time.sleep(sleep_for)
            continue
        raise
if last_exc is not None:
    raise last_exc
print("Uploaded artifacts to", repo_id)
PY
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Launch AdaptShield GRPO refinement on Hugging Face Jobs")
    parser.add_argument("--runs-repo", required=True)
    parser.add_argument("--runs-repo-type", default="model", choices=["dataset", "model"])
    parser.add_argument("--skip-create", action="store_true")
    parser.add_argument("--allow-cross-namespace", action="store_true")
    parser.add_argument("--repo-url", default=None)
    parser.add_argument("--source-repo", required=True, help="Repo containing SFT artifacts.")
    parser.add_argument("--source-repo-type", default="model", choices=["dataset", "model"])
    parser.add_argument("--source-subdir", default="sft_worldsplit_1_5b", help="Subdirectory containing the SFT output.")
    parser.add_argument("--model", default="1.5b", choices=list(MODEL_CHOICES))
    parser.add_argument("--flavor", default="l4x1")
    parser.add_argument("--timeout", default="6h")
    parser.add_argument("--prompt-bank-episodes", type=int, default=120)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--prompt-bank-hard-multiplier", type=int, default=3)
    parser.add_argument("--prompt-bank-borderline-bonus", type=int, default=2)
    parser.add_argument("--grpo-epochs", type=int, default=1)
    parser.add_argument("--num-generations", type=int, default=2)
    parser.add_argument("--per-device-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--heldout-seed", type=int, default=314)
    parser.add_argument("--output-subdir", default="grpo_worldsplit_1_5b")
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("No Hugging Face token found. Run `hf auth login` first.")

    repo_url = args.repo_url or infer_repo_url()
    api = HfApi(token=token)
    validate_repo_access(api, args.runs_repo, args.runs_repo_type, args.skip_create, args.allow_cross_namespace)
    validate_repo_access(api, args.source_repo, args.source_repo_type, True, args.allow_cross_namespace)
    validate_source_artifacts(api, args.source_repo, args.source_repo_type, args.source_subdir)
    if not args.skip_create:
        _retry_hf_call(api.create_repo, repo_id=args.runs_repo, repo_type=args.runs_repo_type, private=True, exist_ok=True)

    command = build_command(args=args, repo_url=repo_url, output_subdir=args.output_subdir)
    job = _retry_hf_call(
        run_job,
        image=DEFAULT_IMAGE,
        command=["bash", "-lc", command],
        flavor=args.flavor,
        timeout=args.timeout,
        namespace=repo_namespace(args.runs_repo),
        env={
            "RUNS_REPO": args.runs_repo,
            "RUNS_REPO_TYPE": args.runs_repo_type,
        },
        secrets={"HF_TOKEN": token},
    )

    print("Job launched successfully.")
    print(f"Job ID: {job.id}")
    print(f"Job URL: {job.url}")
    print(f"Artifacts repo: {args.runs_repo}")
    print(f"Artifacts path: {args.output_subdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
