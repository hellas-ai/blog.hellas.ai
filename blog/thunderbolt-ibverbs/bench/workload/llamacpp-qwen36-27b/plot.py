#!/usr/bin/env python3
"""Plot llama.cpp-vulkan throughput for Qwen3.6-27B Q4_K_M on strix-1.

Reads averages.csv (deduplicated 2-repeat means) and writes
tps_vs_concurrency.svg next to it.

Single-host run (no networking) — this is workload characterization, not an
RDMA comparison. The slot count sp=8 is fixed for the chart (all
concurrencies have data at sp=8).
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BENCH_DIR = Path(__file__).parent

MODE_COLOR = {"baseline": "#1f77b4"}
MODE_LABEL = {"baseline": "baseline"}
SP_FIXED = 8


def load_averages():
    rows = []
    with (BENCH_DIR / "averages.csv").open() as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "mode": r["mode"],
                    "sp": int(r["server_parallel"]),
                    "c": int(r["concurrency"]),
                    "agg": float(r["aggregate_pred_tok_s"]),
                    "mean": float(r["mean_predicted_tok_s"]),
                })
            except (ValueError, KeyError):
                continue
    return rows


def main():
    rows = load_averages()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharex=True)

    for mode in ("baseline",):
        pts = sorted(
            (r["c"], r["agg"], r["mean"]) for r in rows
            if r["mode"] == mode and r["sp"] == SP_FIXED
        )
        if not pts:
            continue
        cs, agg, mean = zip(*pts)
        axes[0].plot(cs, agg, color=MODE_COLOR[mode], marker="o", markersize=5,
                     linewidth=1.8, label=MODE_LABEL[mode])
        axes[1].plot(cs, mean, color=MODE_COLOR[mode], marker="o", markersize=5,
                     linewidth=1.8, label=MODE_LABEL[mode])

    for ax in axes:
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 4, 8])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
        ax.tick_params(axis="x", which="minor", bottom=False)
        ax.set_xlabel("Concurrency (in-flight requests)")
        ax.grid(True, alpha=0.25)
        ax.set_ylim(bottom=0)
        ax.legend(loc="upper left", frameon=True)

    axes[0].set_ylabel("Aggregate throughput (tokens/s)")
    axes[0].set_title("Total throughput (sum across users)")
    axes[1].set_ylabel("Per-user throughput (tokens/s)")
    axes[1].set_title("Per-user decode rate (mean)")

    fig.suptitle(
        "llama.cpp-vulkan Qwen3.6-27B Q4_K_M on strix-1, "
        f"sp={SP_FIXED} slots (2026-05-21)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = BENCH_DIR / "tps_vs_concurrency.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
