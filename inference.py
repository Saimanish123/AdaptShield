"""
AdaptShield Inference Script

Single task per run. Emits mandatory [START]/[STEP]/[END] stdout format.
All credentials read from environment — never hardcoded.

Required env vars (injected by evaluator):
    API_KEY:          Evaluator's LiteLLM proxy key (checked first)
    API_BASE_URL:     LLM endpoint
    MODEL_NAME:       Model identifier

Optional env vars:
    HF_TOKEN:         Fallback if API_KEY not set
    ADAPTSHIELD_TASK: Task name (default: direct-triage)
    ENV_BASE_URL:     Environment server URL (default: localhost:7860)
"""

import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional
import urllib.request
import urllib.error

from openai import OpenAI

from adaptshield import AdaptshieldEnv, AdaptShieldAction
from soc_tools import attach_tool_results, investigate_http, summarize_tool_results

# ── Configuration — read from env, NEVER hardcode ──────────────────────────
API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN", "")
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
TASK_NAME    = os.environ.get("ADAPTSHIELD_TASK", "direct-triage")
ENV_BASE_URL = os.environ.get("ENV_BASE_URL",  "http://localhost:7860").rstrip("/")
BENCHMARK    = "adaptshield"
MAX_STEPS    = 25
SUCCESS_THRESHOLD = 0.50
USE_TOOLS_SETTING = os.environ.get("ADAPTSHIELD_USE_TOOLS", "auto").lower()


# ── Mandatory stdout format ────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    ev = error if error else "null"
    print(
        f"[STEP] step={step} action={action} "
        f"reward={reward:.2f} done={str(done).lower()} error={ev}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float,
            rewards: List[float]) -> None:
    rs = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rs}",
        flush=True,
    )


# ── Environment calls ──────────────────────────────────────────────────────
def env_post(path: str, data: Dict) -> Dict:
    url  = f"{ENV_BASE_URL}{path}"
    body = json.dumps(data).encode()
    req  = urllib.request.Request(
        url, data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def obs_to_dict(obs: Any) -> Dict[str, Any]:
    """Convert Pydantic observations from the persistent client to JSON dicts."""
    if hasattr(obs, "model_dump"):
        return obs.model_dump(mode="json")
    return dict(obs)


def build_env_action(parsed: Dict[str, Any], phase: int) -> AdaptShieldAction:
    """Validate model output and fall back to a phase-correct safe action."""
    try:
        return AdaptShieldAction(**parsed)
    except Exception:
        if phase == 1:
            return AdaptShieldAction(
                threat_type="brute_force",
                confidence=0.5,
                target_node="auth_service",
                recommended_action="monitor",
                reasoning="validated fallback",
            )
        return AdaptShieldAction(
            action="monitor",
            target_node="auth_service",
            reasoning="validated fallback",
        )


# ── Score computation — strictly (0.01, 0.99) ─────────────────────────────
def safe_score(rewards: List[float], meta: Dict) -> float:
    if "normalized_score" in meta:
        raw = float(meta["normalized_score"])
    elif rewards:
        pos  = sum(r for r in rewards if r > 0.50)
        maxp = len(rewards) * 0.99
        raw  = pos / maxp if maxp > 0 else 0.50
    else:
        raw = 0.50
    return max(0.01, min(0.99, raw))


# ── System prompts ─────────────────────────────────────────────────────────
PHASE1_SYS = textwrap.dedent("""
    You are a Threat Analyst for a 4-node enterprise network.
    Analyze the SIEM metrics and alerts. Identify the threat type.

    Attack strategies: brute_force, lateral_movement, exfiltration, supply_chain, benign
    If SOC tool evidence is provided, use it to update your belief before classifying.

    Respond ONLY with valid JSON:
    {"threat_type":"...","confidence":0.0,"target_node":"...","recommended_action":"...","reasoning":"..."}

    Nodes: auth_service, payment_service, database, api_gateway
    Actions: rate_limit, isolate, honeypot, patch, monitor
""").strip()

PHASE2_SYS = textwrap.dedent("""
    You are a Tactical Executor. Act on the threat assessment provided.
    You cannot see raw network data. Use the analyst assessment plus any SOC tool trace.

    rate_limit=throttle traffic, isolate=take offline, honeypot=redirect attacker,
    patch=fix vulnerability, monitor=observe only

    Respond ONLY with valid JSON:
    {"action":"...","target_node":"...","reasoning":"..."}

    Nodes: auth_service, payment_service, database, api_gateway
""").strip()


def get_action(client: OpenAI, obs: Dict) -> Dict[str, Any]:
    """Call LLM for current phase. Falls back gracefully on parse error."""
    phase = obs.get("phase", 1)

    if phase == 1:
        sys_msg = PHASE1_SYS
        user_msg = "\n".join([
            "Network nodes:",
            json.dumps(obs.get("network_nodes", {}), indent=2),
            "\nAlerts:",
            "\n".join(obs.get("active_alerts", [])),
            "\nSOC tool evidence:",
            summarize_tool_results(obs.get("tool_results", [])),
            "\nHistory:",
            json.dumps(obs.get("history", []), indent=2),
            "\nClassify the threat:",
        ])
        fallback = {
            "threat_type": "brute_force", "confidence": 0.5,
            "target_node": "auth_service", "recommended_action": "monitor",
            "reasoning": "fallback",
        }
    else:
        sys_msg = PHASE2_SYS
        metadata = obs.get("metadata", {}) if isinstance(obs.get("metadata", {}), dict) else {}
        current_turn = int(obs.get("turn", 0) or 0)
        tool_trace = [
            row for row in metadata.get("tool_trace", [])
            if int(row.get("turn", -1)) == current_turn
        ]
        user_msg = "\n".join([
            "Threat assessment from analyst:",
            json.dumps(obs.get("phase1_assessment", {}), indent=2),
            "\nSOC tool trace for this turn:",
            json.dumps(tool_trace, indent=2),
            "\nChoose your defensive action:",
        ])
        fallback = {
            "action": "monitor",
            "target_node": "auth_service",
            "reasoning": "fallback",
        }

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system",  "content": sys_msg},
                {"role": "user",    "content": user_msg},
            ],
            temperature=0.1,
            max_tokens=300,
            stream=False,
        )
        text = (resp.choices[0].message.content or "").strip()

        # Strip markdown fences
        if "```" in text:
            for part in text.split("```"):
                if "{" in part:
                    text = part.strip().lstrip("json").strip()
                    break

        return json.loads(text)

    except Exception as exc:
        print(f"[DEBUG] phase={phase} parse error: {exc}", flush=True)
        return fallback


def should_use_tools(task_name: str) -> bool:
    if USE_TOOLS_SETTING in ("1", "true", "yes", "on"):
        return True
    if USE_TOOLS_SETTING in ("0", "false", "no", "off"):
        return False
    return task_name == "polymorphic-zero-day"


def run_soc_episode(client: OpenAI, use_tools: bool) -> tuple[List[float], int, Dict[str, Any]]:
    rewards: List[float] = []
    steps_taken = 0

    reset = env_post("/soc/reset", {"task": TASK_NAME})
    session_id = str(reset.get("session_id", ""))
    obs = dict(reset.get("observation", {}))
    done = bool(obs.get("done", False))

    for step in range(1, MAX_STEPS + 1):
        if done:
            break

        tool_results = investigate_http(
            env_base_url=ENV_BASE_URL,
            session_id=session_id,
            obs=obs,
            use_tools=use_tools,
            thorough=True,
        )
        obs_for_model = attach_tool_results(obs, tool_results)
        parsed = get_action(client, obs_for_model)
        action_str = json.dumps(parsed, separators=(",", ":"))
        if len(action_str) > 100:
            action_str = action_str[:97] + "..."

        try:
            action = build_env_action(parsed, phase=int(obs.get("phase", 1)))
            action_payload = action.model_dump(
                mode="json",
                exclude_none=True,
                exclude_defaults=True,
            )
            result = env_post("/soc/step", {"session_id": session_id, "action": action_payload})
            obs = dict(result.get("observation", {}))
            reward = float(result.get("reward", obs.get("reward", 0.0)))
            done = bool(result.get("done", obs.get("done", False)))
            error = None
        except Exception as exc:
            reward = 0.0
            done = True
            error = str(exc)[:80]

        rewards.append(reward)
        steps_taken = step
        log_step(step=step, action=action_str, reward=reward, done=done, error=error)

        if done:
            break

    return rewards, steps_taken, obs


def run_openenv_episode(client: OpenAI) -> tuple[List[float], int, Dict[str, Any]]:
    rewards: List[float] = []
    steps_taken = 0
    obs: Dict[str, Any] = {}

    env = AdaptshieldEnv(base_url=ENV_BASE_URL).sync()
    with env:
        result = env.reset(task_name=TASK_NAME)
        obs = obs_to_dict(result.observation)
        done = bool(result.done or obs.get("done", False))

        for step in range(1, MAX_STEPS + 1):
            if done:
                break

            parsed = get_action(client, obs)
            action_str = json.dumps(parsed, separators=(",", ":"))
            if len(action_str) > 100:
                action_str = action_str[:97] + "..."

            try:
                action = build_env_action(parsed, phase=int(obs.get("phase", 1)))
                sr = env.step(action)
                obs = obs_to_dict(sr.observation)
                reward = float(sr.reward if sr.reward is not None else obs.get("reward", 0.0))
                done = bool(sr.done or obs.get("done", False))
                error = None
            except Exception as exc:
                reward = 0.0
                done = True
                error = str(exc)[:80]

            rewards.append(reward)
            steps_taken = step
            log_step(step=step, action=action_str, reward=reward,
                     done=done, error=error)

            if done:
                break

    return rewards, steps_taken, obs


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.50
    success:     bool        = False
    obs:         Dict        = {}

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        if should_use_tools(TASK_NAME):
            rewards, steps_taken, obs = run_soc_episode(client, use_tools=True)
        else:
            rewards, steps_taken, obs = run_openenv_episode(client)

        score   = safe_score(rewards, obs.get("metadata", {}))
        success = score >= SUCCESS_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] episode error: {exc}", flush=True)
        score = 0.10

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
