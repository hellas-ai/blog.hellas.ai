#!/usr/bin/env python3
"""Plot vLLM Qwen3-0.6B: solo (single host) vs TP=2 4-HCA RDMA.

Reads the *.csv next to this script (the vllm-pair schema) and writes
tps_vs_transport.svg. Two bars — solo vs tp2 RDMA — for total throughput
at concurrency=256 with 256-token outputs.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

BENCH_DIR = Path(__file__).parent

VARIANTS = [
    ("solo",      "TP=1 solo (single host)",      "#7f7f7f"),
    ("usb4_rdma", "TP=2 RDMA 4-HCA (both hosts)", "#d62728"),
]


def load_rows() -> list[dict]:
    out: list[dict] = []
    for csv_path in sorted(BENCH_DIR.glob("*.csv")):
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                if r.get("status") == "ok":
                    out.append(r)
    return out


def to_float(s) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def plot_tps(rows: list[dict]) -> None:
    by_transport: dict[str, dict] = {}
    for r in rows:
        by_transport[r["transport"]] = r

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for key, label, color in VARIANTS:
        r = by_transport.get(key)
        if not r:
            continue
        v = to_float(r.get("total_tps"))
        if v is None:
            continue
        labels.append(label)
        values.append(v)
        colors.append(color)

    bars = ax.barh(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    vmax = max(values) if values else 0
    for bar, val in zip(bars, values):
        ax.text(
            val + vmax * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{val:,.0f} tok/s",
            va="center",
            fontsize=9,
        )
    ax.set_xlim(right=vmax * 1.18)
    ax.invert_yaxis()
    ax.set_xlabel("Total throughput (tokens/s, prompt+output, higher = better)")
    ax.set_title(
        "vLLM Qwen3-0.6B (concurrency=256, 256-tok output) — solo vs TP=2 4-HCA RDMA",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(BENCH_DIR / "tps_vs_transport.svg")
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    if not rows:
        print("no CSV rows found")
        return
    plot_tps(rows)
    print(f"wrote tps_vs_transport.svg ({len(rows)} rows)")


if __name__ == "__main__":
    main()
