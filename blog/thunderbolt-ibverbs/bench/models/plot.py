#!/usr/bin/env python3
"""ELO vs parameter-count chart with 'AI at home' hardware-envelope bands.

Reads models.csv next to this script and emits two SVGs:

  - elo_vs_params.svg          three vertical bands (single GPU, ganged GPUs,
                               ganged machines) at the FP16 fit ceilings.
  - elo_vs_params_quant.svg    same scatter, with the bands widened to the
                               Q4_K_M (~5 bit/param) ceilings, illustrating
                               how aggressive quantization stretches the
                               at-home envelope.

Both charts use total parameter count for the x-axis (memory ceiling — even
for MoE the full weights must be resident). Open-weight models are filled
markers; proprietary (closed-weight) models are hollow / ghosted so they
read as a reference ceiling rather than something a reader can run at home.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter

BENCH_DIR = Path(__file__).parent

# Hardware envelope thresholds — params (B) that fit in each tier.
# Rule of thumb: bytes/param × params ≤ memory_GB × 1e9 → params (B) ≤ memory_GB / bytes_per_param.
# FP16 = 2 bytes/param. Q4_K_M ≈ 5 bits/param ≈ 0.625 bytes/param.
# Accept this is hand-wavy: KV cache + activations add ~20–50% at long context.
TIERS = [
    {
        "label": "single GPU\n(24-32 GB)",
        "color": "#9BBC9F",
        "fp16_max_b": 12,
        "q4_max_b": 40,
    },
    {
        "label": "ganged GPUs\n(milk-crate, 4× 24 GB)",
        "color": "#E8DA9C",
        "fp16_max_b": 48,
        "q4_max_b": 150,
    },
    {
        "label": "ganged machines\n(2× Strix Halo, 256 GB)",
        "color": "#E8B89C",
        "fp16_max_b": 120,
        "q4_max_b": 400,
    },
]


def load_rows():
    rows = []
    with (BENCH_DIR / "models.csv").open() as f:
        for r in csv.DictReader(f):
            try:
                params = float(r["params_b"]) if r["params_b"] else None
                elo = float(r["elo"]) if r["elo"] else None
            except ValueError:
                continue
            rows.append({
                "model": r["model"],
                "params_b": params,
                "elo": elo,
                "family": r["family"],
                "license": r["license"],
            })
    return rows


def family_color(family):
    return {
        "alibaba": "#d62728",
        "deepseek": "#1f77b4",
        "meta": "#2178d4",
        "google": "#4285f4",
        "anthropic": "#cc785c",
        "openai": "#10a37f",
        "xai": "#5e5e5e",
        "moonshot": "#9467bd",
        "zhipu": "#17becf",
        "mistral": "#ff7f0e",
        "microsoft": "#7c41a7",
        "huggingface": "#ff9d00",
        "baidu": "#2932e1",
        "xiaomi": "#ff6700",
    }.get(family, "#7f7f7f")


def draw_chart(rows, *, use_quant: bool, out_path: Path, title: str, subtitle: str):
    fig, ax = plt.subplots(figsize=(11, 6.5))

    # Hardware bands as shaded vertical regions.
    x_left = 0.5
    for i, tier in enumerate(TIERS):
        right = tier["q4_max_b"] if use_quant else tier["fp16_max_b"]
        ax.axvspan(x_left, right, color=tier["color"], alpha=0.45,
                   zorder=-2, linewidth=0)
        ax.text(
            (x_left * right) ** 0.5,                 # geometric-mean centre on log axis
            0.96,
            tier["label"],
            ha="center", va="top", fontsize=8.5, color="#444",
            transform=ax.get_xaxis_transform(),
        )
        x_left = right
    # final "cloud only" band
    ax.axvspan(x_left, 2000, color="#cccccc", alpha=0.35, zorder=-2, linewidth=0)
    ax.text(
        (x_left * 2000) ** 0.5, 0.96, "cloud / very large rigs",
        ha="center", va="top", fontsize=8.5, color="#444",
        transform=ax.get_xaxis_transform(),
    )

    # Scatter the models.
    plotted = [r for r in rows if r["params_b"] and r["elo"]]
    for r in plotted:
        is_open = r["license"] == "open-weight"
        color = family_color(r["family"])
        ax.scatter(
            r["params_b"], r["elo"],
            s=72 if is_open else 60,
            c=color if is_open else "white",
            edgecolors=color,
            linewidths=1.5,
            marker="o",
            zorder=3,
            alpha=0.95 if is_open else 0.65,
        )
        # label
        ax.annotate(
            r["model"], (r["params_b"], r["elo"]),
            xytext=(5, 4), textcoords="offset points",
            fontsize=7.2, color="#222",
        )

    # Closed-weight callout near the top — show as a horizontal "frontier line".
    closed_elos = [r["elo"] for r in rows if r["license"] == "proprietary" and r["elo"]]
    if closed_elos:
        frontier = max(closed_elos)
        ax.axhline(frontier, color="#bbb", linestyle="--", linewidth=1, zorder=1)
        ax.text(
            1800, frontier + 2,
            f"frontier (closed-weight): {int(frontier)}",
            ha="right", fontsize=8, color="#666",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Total parameters (B, log scale)")
    ax.set_ylabel("LMArena ELO")
    ax.set_xlim(0.5, 2000)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xticks([1, 4, 10, 30, 100, 300, 1000])
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.grid(True, which="both", alpha=0.2, zorder=0)
    ax.set_title(title + "\n" + subtitle, fontsize=11, color="#222", pad=14,
                 loc="left")

    # Legend: open vs closed
    legend = [
        mpatches.Patch(color="#999", label="filled = open-weight"),
        mpatches.Patch(facecolor="white", edgecolor="#999", label="hollow = proprietary"),
    ]
    ax.legend(handles=legend, loc="lower right", fontsize=8, frameon=True)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Wrote {out_path}")


def main():
    rows = load_rows()
    draw_chart(
        rows,
        use_quant=False,
        out_path=BENCH_DIR / "elo_vs_params.svg",
        title="Open-weight LLM frontier vs 'AI at home' hardware envelope (FP16)",
        subtitle="data: arena.ai/leaderboard/text (top 40) + HF cards; hardware bands assume FP16 weights; "
                 "MoE rows charted by total params (memory ceiling)",
    )
    draw_chart(
        rows,
        use_quant=True,
        out_path=BENCH_DIR / "elo_vs_params_quant.svg",
        title="Same data, hardware envelope at Q4_K_M (~5 bits/param)",
        subtitle="Q4_K_M quantization stretches each tier ~4× to the right — "
                 "frontier 100-400 B models become runnable on a 2-node Strix Halo cluster",
    )


if __name__ == "__main__":
    main()
