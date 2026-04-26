"""Render the headline benchmark chart for README.

Produces a clean grouped bar chart of held-out evaluation scores
(tool-aware baseline / SFT / GRPO) across the three difficulty tiers.

Numbers are pulled directly from
https://huggingface.co/SaiManish123/Janus benchmark tables and are
identical to the values in README.md so the figure stays in sync.

Run: python assets/_make_headline_chart.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-adaptshield")

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent / "headline_results.png"

tasks = ["direct-triage\n(easy)", "dual-pivot\n(medium)", "polymorphic-zero-day\n(hard)"]
tool_baseline = [0.990, 0.640, 0.180]
sft_heldout = [0.990, 0.825, 0.930]
grpo_heldout = [0.990, 0.825, 0.902]

x = np.arange(len(tasks))
width = 0.26

fig, ax = plt.subplots(figsize=(9.5, 4.6), dpi=150)

c_tool = "#9aa0a6"
c_sft = "#1f6feb"
c_grpo = "#d63b2f"

b1 = ax.bar(x - width, tool_baseline, width, label="Tool-aware baseline", color=c_tool, edgecolor="white", linewidth=0.6)
b2 = ax.bar(x,         sft_heldout,  width, label="SFT (held-out)",      color=c_sft,  edgecolor="white", linewidth=0.6)
b3 = ax.bar(x + width, grpo_heldout, width, label="GRPO (held-out)",     color=c_grpo, edgecolor="white", linewidth=0.6)

for bars in (b1, b2, b3):
    ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=9, color="#333")

ax.set_ylim(0, 1.08)
ax.set_yticks(np.arange(0, 1.01, 0.2))
ax.set_ylabel("Mean score (0.01–0.99 grader)", fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(tasks, fontsize=10)
ax.set_title(
    "AdaptShield held-out evaluation · Qwen2.5-1.5B · 50 deterministic seeds / task",
    fontsize=11.5, pad=12, color="#222",
)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_color("#cccccc")
ax.spines["bottom"].set_color("#cccccc")
ax.tick_params(colors="#555")
ax.yaxis.grid(True, color="#eeeeee", linewidth=0.8)
ax.set_axisbelow(True)

ax.annotate(
    "5.0× lift on the only task that\nactually requires adaptation",
    xy=(2 + width, grpo_heldout[2]),
    xytext=(2 - 0.15, 0.45),
    fontsize=9, color="#444",
    arrowprops=dict(arrowstyle="->", color="#888", lw=0.9, connectionstyle="arc3,rad=-0.2"),
)

ax.legend(
    loc="lower left", frameon=False, fontsize=9.5, ncol=3,
    bbox_to_anchor=(0.0, -0.22),
)

plt.tight_layout()
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(f"wrote {OUT}")
