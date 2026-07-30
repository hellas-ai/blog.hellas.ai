#!/usr/bin/env python3
"""Plot vLLM Gemma-4-26B-A4B-it: TP=1 local vs TP=2 RDMA on strix-1+strix-2.

Reads summary.tsv files from each variant subdirectory and writes
tps_vs_concurrency.svg next to this script.
"""

import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt

BENCH_DIR = Path(__file__).parent

# Insertion order = bar order, left to right
VARIANTS = {
    "tp1-local-eager":       {"label": "TP=1 local (single host)",   "color": "#7f7f7f"},
    "tp2-rdma-cold":         {"label": "TP=2 RDMA (cold)",           "color": "#1f77b4"},
    "tp2-rdma-lowcon-eager": {"label": "TP=2 RDMA (eager warm-ish)", "color": "#ff7f0e"},
}

CONCURRENCY_RE = re.compile(r"-c(\d+)\.json$")


def load():
    """variant → concurrency → metrics dict; only rows where all reqs OK."""
    out = {}
    for variant in VARIANTS:
        tsv = BENCH_DIR / variant / "summary.tsv"
        if not tsv.exists():
            continue
        per_c = {}
        with tsv.open() as f:
            for r in csv.DictReader(f, delimiter="\t"):
                m = CONCURRENCY_RE.search(r["file"])
                if not m:
                    continue
                c = int(m.group(1))
                try:
                    completed = int(r["completed"])
                    failed = int(r["failed"])
                except (ValueError, KeyError):
                    continue
                if failed > 0 or completed == 0:
                    continue  # exclude crashed runs from the chart
                try:
                    per_c[c] = {
                        "output_tps": float(r["output_tok/s"]),
                        "total_tps": float(r["total_tok/s"]),
                        "ttft_ms": float(r["mean_ttft_ms"]),
                    }
                except (ValueError, KeyError):
                    continue
        out[variant] = per_c
    return out


def main():
    data = load()
    concurrencies = sorted({c for v in data.values() for c in v.keys()})

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    x = list(range(len(concurrencies)))
    bar_w = 0.25
    n = len(VARIANTS)

    for i, (variant, props) in enumerate(VARIANTS.items()):
        ys = [data.get(variant, {}).get(c, {}).get("output_tps") for c in concurrencies]
        offsets = [px + (i - (n - 1) / 2) * bar_w for px in x]
        # gracefully drop missing bars
        valid_off, valid_y = zip(*[(o, y) for o, y in zip(offsets, ys) if y is not None])
        bars = ax.bar(valid_off, valid_y, bar_w,
                      color=props["color"], label=props["label"],
                      edgecolor="black", linewidth=0.4)
        for bar, val in zip(bars, valid_y):
            ax.annotate(f"{val:.1f}", (bar.get_x() + bar.get_width() / 2, val),
                        xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=8, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels([f"c={c}" for c in concurrencies])
    ax.set_xlabel("Concurrency (in-flight requests)")
    ax.set_ylabel("Output throughput (tokens/s)")
    ax.set_title(
        "vLLM Gemma-4-26B-A4B-it (~52 GB bf16) on strix-1 ↔ strix-2 (2026-05-21)"
    )
    ax.text(
        0.5, -0.18,
        "Note: TP=2 RDMA cold also tried c=8 — 3 of 8 requests failed, "
        "output_tok/s collapsed to 0.03 (excluded from chart).",
        transform=ax.transAxes, ha="center", va="top", fontsize=8, color="#555",
    )
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(bottom=0)

    out = BENCH_DIR / "tps_vs_concurrency.svg"
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
