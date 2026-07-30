#!/usr/bin/env python3
"""Plots for the 2-node Strix-Halo Gemma 3 27B LoRA FSDP one-step benches.

One row per (transport) in the CSV; the headline is wall-clock for a
single train step at batch=1, max_length=128, lora.  Emits two SVGs
next to this script:

  - runtime_vs_transport.svg : train_runtime_s per transport, bar plot.
  - bw_vs_transport.svg      : effective per-direction throughput per
                               transport, derived the same way as
                               bench/finetune/plot.py.
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

# Matches the TX ring size assumed in bench/finetune/plot.py.
RDMA_FRAME_BYTES = 4096


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


def to_int(s) -> int | None:
    if s is None or s == "":
        return None
    try:
        return int(float(s))
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

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
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
        "Gemma 3 27B LoRA FSDP — 1-step wall time on 2× Strix Halo",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(BENCH_DIR / "runtime_vs_transport.svg")
    plt.close(fig)


def bw_plot(rows: list[dict]) -> None:
    grouped: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        rt = to_float(r.get("training.train_runtime_s"))
        if rt is None:
            continue
        t = r.get("row.transport")
        if t is None:
            continue
        bytes_total = 0
        if t == "eth":
            rx = to_int(r.get("delta.master.eth.br0.lan.rx_bytes")) or 0
            tx = to_int(r.get("delta.master.eth.br0.lan.tx_bytes")) or 0
            bytes_total = max(rx, tx)
        else:
            # Sum all per-rail tx_completed packets we recognise.
            for k, v in r.items():
                if not k.startswith("delta.master.dfs.") or "data_tx_completed" not in k:
                    continue
                n = to_int(v)
                if n:
                    bytes_total += n * RDMA_FRAME_BYTES
        if bytes_total == 0:
            continue
        gbps = (bytes_total * 8) / (rt * 1e9)
        grouped[t].append(gbps)

    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    transports = [t for t in TRANSPORT_ORDER if t in grouped]
    vals = [sum(grouped[t]) / len(grouped[t]) for t in transports]
    colors = [TRANSPORT_COLOR.get(t, "#7f7f7f") for t in transports]
    _barh(
        ax,
        [TRANSPORT_LABEL.get(t, t) for t in transports],
        vals,
        colors,
        lambda v: f"{v:.1f} Gb/s",
    )
    ax.set_xlabel("Effective sustained per-direction throughput (Gb/s, app payload)")
    ax.set_title(
        "Gemma 3 27B LoRA FSDP — interconnect throughput on 2× Strix Halo",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(BENCH_DIR / "bw_vs_transport.svg")
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    if not rows:
        print("no CSV rows found")
        return
    runtime_plot(rows)
    bw_plot(rows)
    print(f"wrote runtime_vs_transport.svg, bw_vs_transport.svg ({len(rows)} rows)")


if __name__ == "__main__":
    main()
