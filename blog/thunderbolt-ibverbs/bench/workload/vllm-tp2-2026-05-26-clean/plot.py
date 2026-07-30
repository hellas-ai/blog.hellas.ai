#!/usr/bin/env python3
"""Solo vs TP=2 (TCP vs RDMA) vLLM throughput at clean 20G link, wifi off.

Each `vllm-pair-*.csv` here is one harness run producing 1-2 rows:
  - a solo baseline (single-host, transport="solo")
  - a TP=2 row using either transport="lan_tcp" or transport="usb4_rdma"

We pivot across CSVs to land at a grouped bar chart of
total_tps per (model, transport) at concurrency=256.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

BENCH_DIR = Path(__file__).parent

TRANSPORT_ORDER = ["solo", "usb4_rdma", "lan_tcp"]
TRANSPORT_LABEL = {
    "solo": "solo (single host, no TP)",
    "usb4_rdma": "TP=2 over usb4_rdma (4-HCA)",
    "lan_tcp": "TP=2 over 2.5G LAN (TCP)",
}
TRANSPORT_COLOR = {
    "solo": "#7f7f7f",
    "usb4_rdma": "#1f77b4",
    "lan_tcp": "#2ca02c",
}

MODEL_ORDER = ["unsloth/Meta-Llama-3.1-8B-Instruct", "Qwen/Qwen3-0.6B"]
MODEL_LABEL = {
    "unsloth/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-0.6B": "Qwen3-0.6B",
}


def load_rows():
    rows = []
    for csv_path in sorted(BENCH_DIR.glob("vllm-pair-*.csv")):
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                if r.get("status") != "ok":
                    continue
                try:
                    rows.append({
                        "model": r["model"],
                        "transport": r["transport"],
                        "concurrency": int(r["concurrency"]),
                        "total_tps": float(r["total_tps"]),
                    })
                except (ValueError, KeyError):
                    pass
    return rows


def main():
    rows = load_rows()
    # average duplicates (e.g. solo measured twice when pairing two transports)
    grouped = defaultdict(list)
    for r in rows:
        if r["concurrency"] != 256:
            continue
        grouped[(r["model"], r["transport"])].append(r["total_tps"])

    models = [m for m in MODEL_ORDER if any(k[0] == m for k in grouped)]
    transports = [t for t in TRANSPORT_ORDER
                  if any(k[1] == t for k in grouped if k[0] in models)]

    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    n = len(transports)
    bar_w = 0.25
    x = list(range(len(models)))

    for i, t in enumerate(transports):
        ys = []
        for m in models:
            vals = grouped.get((m, t), [])
            ys.append(sum(vals) / len(vals) if vals else 0.0)
        offsets = [px + (i - (n - 1) / 2) * bar_w for px in x]
        bars = ax.bar(offsets, ys, bar_w,
                      color=TRANSPORT_COLOR[t],
                      label=TRANSPORT_LABEL[t],
                      edgecolor="black", linewidth=0.4)
        for bar, val in zip(bars, ys):
            if val > 0:
                ax.annotate(f"{val:.0f}",
                            (bar.get_x() + bar.get_width() / 2, val),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", fontsize=8, color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABEL.get(m, m) for m in models])
    ax.set_ylabel("End-to-end throughput (total tokens/s)")
    ax.set_title(
        "vLLM TP=2 vs solo at concurrency 256, clean 20 Gb/s link (wifi off)\n"
        "strix-1 ↔ strix-2 (2026-05-26)",
        fontsize=10,
    )
    ax.grid(True, axis="y", alpha=0.25)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    out = BENCH_DIR / "tps_solo_vs_tp2.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
