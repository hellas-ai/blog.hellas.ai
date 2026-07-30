#!/usr/bin/env python3
"""vLLM MiniMax-M2.7-AWQ (G32, STRIX-2H tune) distributed across 2x
Strix Halo, TP=2.

Two charts:

  - tps_by_transport.svg : single-point bar chart from the n=128
    random64x16 workload (apples-to-apples 'big batch' point) —
    TBnet 320 tps vs RDMA-4HCA 344 tps.
  - tps_vs_batch.svg : line chart from the transport-screen sweep on
    the random64x128 workload, sweeping max_num_seqs in
    {1, 2, 4, 8}. The interesting story: RDMA wins by ~30% at seq=1
    and only ~5% at seq=8 — at low batch each request is
    interconnect-sensitive, at high batch compute amortizes it away.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BENCH_DIR = Path(__file__).parent

TRANSPORT_ORDER = ["tbnet", "rdma-4hca"]
TRANSPORT_LABEL = {
    "tbnet":     "TCP over Thunderbolt (TBnet)",
    "rdma-4hca": "Native 4-HCA RDMA (this work)",
}
TRANSPORT_COLOR = {
    "tbnet":     "#2ca02c",
    "rdma-4hca": "#d62728",
}


def load_rows(pattern):
    out = []
    for csv_path in sorted(BENCH_DIR.glob(pattern)):
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                out.append(r)
    return out


def plot_tps_by_transport():
    """Single-point bar chart at the random64x16 n=128 workload."""
    rows = load_rows("2026-05-27-strix-strix-vllm-minimax-m2-tp2-n128.csv")
    best: dict[str, float] = defaultdict(float)
    for r in rows:
        tps = float(r["tokens_per_s"])
        if tps > best[r["transport"]]:
            best[r["transport"]] = tps

    transports = [t for t in TRANSPORT_ORDER if t in best]
    values = [best[t] for t in transports]
    colors = [TRANSPORT_COLOR.get(t, "#7f7f7f") for t in transports]
    labels = [TRANSPORT_LABEL.get(t, t) for t in transports]

    fig, ax = plt.subplots(figsize=(8.5, 3.4))
    bars = ax.barh(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    vmax = max(values) if values else 0
    for bar, val in zip(bars, values):
        ax.text(
            val + vmax * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.0f} tok/s",
            va="center",
            fontsize=10,
            fontweight="bold",
        )
    ax.set_xlim(right=vmax * 1.18)
    ax.invert_yaxis()
    ax.set_xlabel("Aggregate output throughput (tokens/s, higher = better)")
    ax.set_title(
        "vLLM MiniMax-M2.7 (AWQ, ~230 B MoE / ~10 B active) TP=2\n"
        "on 2× Strix Halo, random 64-token prompts × 16-token gens, n=128",
        fontsize=10,
    )
    ax.grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    out = BENCH_DIR / "tps_by_transport.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_tps_vs_batch():
    """Line chart: throughput vs max_num_seqs ('batch' size), per transport."""
    rows = load_rows("2026-05-27-strix-strix-vllm-minimax-m2-tp2-batch-sweep.csv")
    by_transport = defaultdict(list)
    for r in rows:
        by_transport[r["transport"]].append(
            (int(r["max_num_seqs"]), float(r["tokens_per_s"])))

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for t in TRANSPORT_ORDER:
        if t not in by_transport:
            continue
        pts = sorted(by_transport[t])
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys,
                color=TRANSPORT_COLOR[t],
                marker="o", markersize=6, linewidth=2,
                label=TRANSPORT_LABEL[t])
        for x, y in zip(xs, ys):
            ax.annotate(
                f"{y:.1f}", (x, y),
                xytext=(6, -2 if t == "rdma-4hca" else 8),
                textcoords="offset points",
                fontsize=8, color=TRANSPORT_COLOR[t], fontweight="bold",
            )

    # Improvement % annotations
    rdma = dict(by_transport.get("rdma-4hca", []))
    tbnet = dict(by_transport.get("tbnet", []))
    for seqs in sorted(set(rdma) & set(tbnet)):
        improvement = (rdma[seqs] / tbnet[seqs] - 1) * 100
        ax.annotate(
            f"+{improvement:.0f}%",
            (seqs, (rdma[seqs] + tbnet[seqs]) / 2),
            xytext=(-32, -3), textcoords="offset points",
            fontsize=9, color="#555", ha="right", va="center",
            fontstyle="italic",
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlabel("max_num_seqs (vLLM batch ceiling, log scale)")
    ax.set_ylabel("Aggregate output throughput (tokens/s)")
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title(
        "vLLM MiniMax-M2.7 TP=2 on 2× Strix Halo, random 64×128 — "
        "throughput vs batch, by transport\n"
        "(RDMA wins ~30% at batch 1 and ~5% at batch 8 — interconnect "
        "matters most at low batch where compute can't hide it)",
        fontsize=10,
    )
    ax.legend(loc="upper left", title="Transport", frameon=True)

    fig.tight_layout()
    out = BENCH_DIR / "tps_vs_batch.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    plot_tps_by_transport()
    plot_tps_vs_batch()


if __name__ == "__main__":
    main()
