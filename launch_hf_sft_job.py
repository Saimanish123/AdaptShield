#!/usr/bin/env python3
"""Launch an AdaptShield SFT training run on Hugging Face Jobs."""

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


def validate_artifact_repo(
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
            f"Authenticated HF account appears to be '{username}', but artifacts repo is under '{owner}'. "
            "Use a repo under the same namespace or pass --allow-cross-namespace only if you are certain "
            "this token has write access there."
        )
        if not allow_cross_namespace:
            raise RuntimeError(message)
        print(f"Warning: {message}")

    if skip_create:
        try:
            _retry_hf_call(api.repo_info, repo_id=repo_id, repo_type=repo_type)
        except RepositoryNotFoundError as exc:
            raise RuntimeError(
                f"Artifacts repo '{repo_id}' ({repo_type}) was not found or is not accessible "
                "with the current token. Create it manually under the correct namespace or use "
                "a repo you definitely own before launching the job."
            ) from exc
        except HfHubHTTPError as exc:
            raise RuntimeError(
                f"Could not verify artifacts repo '{repo_id}' ({repo_type}) before launch: {exc}"
            ) from exc


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


def build_command(args: argparse.Namespace, repo_url: str, output_subdir: str) -> str:
    dataset_path = "/workspace/adaptshield/data/adaptshield_sft_worldsplit.jsonl"
    output_path = f"/workspace/adaptshield/checkpoints/{output_subdir}"
    summary_path = "/workspace/adaptshield/data/adaptshield_sft_worldsplit.summary.json"

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
pip install matplotlib unsloth trl accelerate bitsandbytes huggingface_hub
python - <<'PY'
import importlib
for name in ["torch", "transformers", "trl", "unsloth", "peft", "train", "train_sft", "generate_sft_data"]:
    importlib.import_module(name)
print("Dependency smoke check passed.")
PY

python generate_sft_data.py \\
  --task all \\
  --curriculum \\
  --use-tools \\
  --episodes {args.dataset_episodes} \\
  --max-steps {args.max_steps} \\
  --seed {args.seed} \\
  --world-split train \\
  --output {dataset_path}

python train_sft.py \\
  --dataset {dataset_path} \\
  --model {args.model} \\
  --epochs {args.epochs} \\
  --lr {args.lr} \\
  --per-device-batch-size {args.per_device_batch_size} \\
  --gradient-accumulation-steps {args.gradient_accumulation_steps} \\
  --save-steps {args.save_steps} \\
  --heldout-seed {args.heldout_seed} \\
  --train-world-split train \\
  --heldout-world-split eval \\
  --eval-task all \\
  --eval-episodes {args.eval_episodes} \\
  --use-tools \\
  --output {output_path}

python - <<'PY'
import os
import time
from huggingface_hub import HfApi

api = HfApi(token=os.environ["HF_TOKEN"])
repo_id = os.environ["RUNS_REPO"]
repo_type = os.environ["RUNS_REPO_TYPE"]
output_dir = {output_path!r}
summary_path = {summary_path!r}
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
        api.upload_file(
            repo_id=repo_id,
            repo_type=repo_type,
            path_or_fileobj=summary_path,
            path_in_repo=f"{{subdir}}/adaptshield_sft_worldsplit.summary.json",
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
    parser = argparse.ArgumentParser(description="Launch AdaptShield SFT training on Hugging Face Jobs")
    parser.add_argument("--runs-repo", required=True, help="Artifact repo to upload outputs to, e.g. username/adaptshield-runs")
    parser.add_argument("--runs-repo-type", default="dataset", choices=["dataset", "model"], help="Repo type used to store training artifacts.")
    parser.add_argument("--skip-create", action="store_true", help="Skip repo creation and assume the artifacts repo already exists.")
    parser.add_argument("--allow-cross-namespace", action="store_true", help="Allow uploads to a repo owned by a different namespace than the authenticated account.")
    parser.add_argument("--repo-url", default=None, help="Git repo URL to clone inside the HF Job. Defaults to remote.origin.url")
    parser.add_argument("--model", default="1.5b", choices=list(MODEL_CHOICES))
    parser.add_argument("--flavor", default="l4x1", help="HF Jobs hardware flavor, e.g. l4x1, a10g-small, a100-large")
    parser.add_argument("--timeout", default="6h", help="HF Jobs timeout, e.g. 6h")
    parser.add_argument("--dataset-episodes", type=int, default=240)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--per-device-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--save-steps", type=int, default=40)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heldout-seed", type=int, default=314)
    parser.add_argument("--output-subdir", default=None, help="Optional output folder name in the runs dataset repo")
    args = parser.parse_args()

    token = get_token()
    if not token:
        raise RuntimeError("No Hugging Face token found. Run `hf auth login` first.")

    repo_url = args.repo_url or infer_repo_url()
    output_subdir = args.output_subdir or f"sft_worldsplit_{args.model.replace('.', '_')}"

    api = HfApi(token=token)
    validate_artifact_repo(
        api,
        args.runs_repo,
        args.runs_repo_type,
        args.skip_create,
        args.allow_cross_namespace,
    )
    if not args.skip_create:
        _retry_hf_call(api.create_repo, repo_id=args.runs_repo, repo_type=args.runs_repo_type, private=True, exist_ok=True)

    command = build_command(args=args, repo_url=repo_url, output_subdir=output_subdir)
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
    print(f"Artifacts path: {output_subdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
