#!/usr/bin/env python3
"""Plot Gemma 3 12B FULL FSDP 1-step wall time on 2x Strix Halo.

Reads the *.csv next to this script and writes runtime_vs_transport.svg.
Mirrors bench/finetune/gemma3-27b/ but with only the two transports the
full-FT compare run actually executed.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

BENCH_DIR = Path(__file__).parent

TRANSPORT_ORDER = ["eth", "rdma", "rdma-4hca"]
TRANSPORT_LABEL = {
    "eth": "TCP over br0.lan (Ethernet)",
    "rdma": "RDMA (single-HCA, new module)",
    "rdma-4hca": "RDMA (4-HCA, new module per-port)",
}
TRANSPORT_COLOR = {
    "eth": "#2ca02c",
    "rdma": "#1f77b4",
    "rdma-4hca": "#d62728",
}


def load_rows() -> list[dict]:
    out: list[dict] = []
    for csv_path in sorted(BENCH_DIR.glob("*.csv")):
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                out.append(r)
    return out


def to_float(s) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _barh(ax, labels, values, colors, value_fmt):
    bars = ax.barh(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    vmax = max(values) if values else 0
    for bar, val in zip(bars, values):
        ax.text(
            val + vmax * 0.012,
            bar.get_y() + bar.get_height() / 2,
            value_fmt(val),
            va="center",
            fontsize=9,
        )
    ax.set_xlim(right=vmax * 1.18)
    ax.invert_yaxis()


def runtime_plot(rows: list[dict]) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        rt = to_float(r.get("training.train_runtime_s"))
        t = r.get("row.transport")
        if rt is None or t is None:
            continue
        grouped[t].append(rt)

    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    transports = [t for t in TRANSPORT_ORDER if t in grouped]
    times = [sum(grouped[t]) / len(grouped[t]) for t in transports]
    colors = [TRANSPORT_COLOR.get(t, "#7f7f7f") for t in transports]
    _barh(
        ax,
        [TRANSPORT_LABEL.get(t, t) for t in transports],
        times,
        colors,
        lambda v: f"{v:.0f} s",
    )
    ax.set_xlabel("Wall clock (s, 1 step, lower = better)")
    ax.set_title(
        "Gemma 3 12B FULL FSDP — 1-step wall time on 2× Strix Halo",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(BENCH_DIR / "runtime_vs_transport.svg")
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    if not rows:
        print("no CSV rows found")
        return
    runtime_plot(rows)
    print(f"wrote runtime_vs_transport.svg ({len(rows)} rows)")


if __name__ == "__main__":
    main()
