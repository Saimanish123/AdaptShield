# AdaptShield — Comprehensive Project Deep Dive
### A Complete Technical, Architectural, and Strategic Reference

---

> **How to read this document:** This is a living reference document covering every file, every connection, every design decision, every strength, every weakness, and the full pitch for the AdaptShield project. It is structured from high-level concepts down to line-level code logic.

---

## PART 1 — Executive Summary & Project Identity

---

### 1.1 What Is AdaptShield?

AdaptShield is a **two-phase agentic cybersecurity reinforcement learning (RL) environment** designed to train Large Language Models (LLMs) to defend a simulated enterprise network against a scripted, adaptive attacker.

It is **not** a traditional RL gym. It is an **OpenEnv-compliant** environment — meaning it follows the OpenEnv protocol for agentic AI evaluation, with a standardized `reset()` / `step()` interface, a FastAPI HTTP server, a Pydantic-typed action/observation contract, and a fully deterministic reward function.

The core idea is this: **current AI security tools fail in production because attackers adapt.** AdaptShield is the first RL environment that forces an AI agent to simultaneously:

1. **Classify** an ongoing attack from ambiguous network telemetry (Phase 1 — Threat Analyst)
2. **Execute** a defense based only on its own classification — not the raw data (Phase 2 — Tactical Executor)
3. **Adapt in real time** as the attacker changes strategy mid-episode
4. **Use investigative tools** (SIEM log search, EDR, CMDB, vulnerability lookup) to gather evidence before acting
5. **Respect operational constraints** — service criticality, blast radius, SLA priorities

This makes AdaptShield qualitatively harder than existing cybersecurity RL benchmarks, which typically give agents full ground truth and a flat action space with no adversarial adaptation.

---

### 1.2 Core Innovation Summary

| Dimension | What AdaptShield Does Differently |
|---|---|
| **Dual-role agent** | Agent plays Threat Analyst AND Tactical Executor in alternating phases |
| **Information asymmetry** | Phase 2 is deliberately blind to raw network state |
| **Adaptive attacker** | Scripted adversary shifts strategy mid-episode on the hard task |
| **Stage escalation** | Attack worsens (recon → exploit → exfiltration) if agent fails to act |
| **SOC tooling layer** | 4 investigative tools return partial, stateful, non-ground-truth evidence |
| **Operational impact** | Reward penalizes unnecessary service disruption, not just wrong actions |
| **Mission awareness** | Each task has a mission profile (SLA priority, risk tolerance) that modulates reward |
| **Catastrophic failure** | Database exfiltration completion ends the episode with severe penalty |
| **Fully deterministic grading** | No LLM-as-judge, no NLP, pure Python strategy matching |
| **OpenEnv compliant** | Works with the OpenEnv evaluator, Docker-deployable, HuggingFace Spaces-ready |

---

### 1.3 The Problem Statement (Why AdaptShield Exists)

Modern enterprise security operations centers (SOCs) face a fundamental challenge: **attackers adapt faster than defenders can retrain models**. Current AI-based security tools:

- Are trained on static labeled datasets (no real-time adaptation)
- Receive full ground-truth telemetry (unrealistic in production)
- Are evaluated on classification accuracy, not operational decision quality
- Ignore the downstream blast radius of defensive actions
- Cannot handle mid-episode strategy shifts by the adversary

AdaptShield directly addresses all five gaps by building a **closed-loop, episodic, multi-turn RL environment** where:

- The attacker is a **scripted Python adversary** (never an LLM) that progresses through stages and shifts strategy
- The agent operates with **partial observability** (Phase 2 gets no raw state)
- Grading is **decision-quality based**, not just classification accuracy
- Operational impact (criticality, dependencies, SLA) **modulates the reward**
- The hardest task requires **multi-tool evidence fusion** for full credit

---

### 1.4 High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         AGENT (LLM / Rule)                      │
│                                                                 │
│    ┌─────────────────────┐    ┌──────────────────────────────┐  │
│    │  Phase 1             │    │  Phase 2                     │  │
│    │  Threat Analyst      │───▶│  Tactical Executor           │  │
│    │  (reads raw SIEM)    │    │  (blind to raw state)        │  │
│    └─────────────────────┘    └──────────────────────────────┘  │
└───────────────────┬─────────────────────────┬───────────────────┘
                    │ Phase1Action             │ Phase2Action
                    ▼                         ▼
┌─────────────────────────────────────────────────────────────────┐
│              AdaptShieldEnvironment (Python class)              │
│                                                                 │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────────────┐  │
│  │ AttackerEngine│  │  Grader       │  │  SOC Tool Surface   │  │
│  │ (scripted    │  │ (deterministic│  │  log_search         │  │
│  │  adversary)  │  │  reward)      │  │  cmdb_lookup        │  │
│  └──────────────┘  └───────────────┘  │  edr_status         │  │
│                                       │  vuln_lookup        │  │
│  ┌──────────────┐  ┌───────────────┐  └─────────────────────┘  │
│  │ Scenarios    │  │ Active Defense│                            │
│  │ (tasks,      │  │ TTL Registry  │                            │
│  │  sys prompts)│  │               │                            │
│  └──────────────┘  └───────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ (HTTP via FastAPI)
┌─────────────────────────────────────────────────────────────────┐
│                     app.py — FastAPI Server                     │
│                                                                 │
│  /reset   /step   /state                                        │
│  /soc/reset   /soc/step                                         │
│  /tools/log_search   /tools/cmdb_lookup                         │
│  /tools/edr_status   /tools/vuln_lookup                         │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
         inference.py              train.py
         (LLM agents,              (LoRA fine-tuning:
          SOC HTTP calls)           PG + GRPO paths)
```

---

### 1.5 Repository Layout

```
adaptshield/                        ← repo root
│
├── adaptshield/                    ← Python package
│   ├── __init__.py                 ← Public exports
│   ├── client.py                   ← WebSocket/HTTP client (AdaptshieldEnv)
│   ├── models.py                   ← Pydantic data models
│   ├── README.md                   ← HuggingFace Space card + usage docs
│   ├── openenv.yaml                ← OpenEnv spec declaration
│   ├── pyproject.toml              ← Package build config
│   ├── Dockerfile                  ← Docker build (HF Spaces + evaluator)
│   ├── uv.lock                     ← Locked dependency manifest
│   │
│   └── server/                     ← Core environment server
│       ├── __init__.py             ← Server package init
│       ├── app.py                  ← FastAPI server + SOC tool endpoints
│       ├── adaptshield_environment.py  ← Main environment class (967 lines)
│       ├── attacker.py             ← Scripted adversary engine (285 lines)
│       ├── grader.py               ← Deterministic reward function (465 lines)
│       ├── scenarios.py            ← Task configs + observation builders
│       └── requirements.txt        ← Server-side deps
│
├── train.py                        ← GPU training harness (PG + GRPO)
├── train_smoke.py                  ← Dependency-free training smoke test
├── inference.py                    ← LLM inference loop (SOC + OpenEnv paths)
├── baseline.py                     ← Rule-based deterministic baseline
├── soc_tools.py                    ← Shared SOC investigation helpers
├── smoke_test.py                   ← Quick import + flow verification
├── eval_tasks.py                   ← Task evaluation runner
├── plot_training.py                ← Training curve visualization
│
├── tests/
│   └── test_regression.py          ← Regression test suite (unittest)
│
├── TRAINING.md                     ← Training guide
├── WORLD_MODELING.md               ← SOC tool layer documentation
└── PROJECT_DEEP_DIVE.md            ← (Previous notes)
```

---

### 1.6 Technology Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Environment core** | Python 3.10+, pure stdlib | Attacker, grader, scenarios |
| **Type system** | Pydantic v2 | Action/observation validation, Enum enforcement |
| **HTTP server** | FastAPI + Uvicorn | OpenEnv-compliant REST API |
| **OpenEnv protocol** | openenv-core ≥ 0.2.2 | Standardized env interface |
| **LLM inference** | openai SDK (via LiteLLM proxy) | Agent LLM calls |
| **Training** | unsloth + TRL (GRPO) + PyTorch | LoRA fine-tuning |
| **Containerization** | Docker (ghcr.io/meta-pytorch/openenv-base) | HF Spaces deployment |
| **Dependency mgmt** | uv (Astral) | Fast locked installs |
| **Testing** | unittest (stdlib) | Regression coverage |

---

*End of Part 1. Continue to Part 2: Data Models & Action/Observation Schema.*
