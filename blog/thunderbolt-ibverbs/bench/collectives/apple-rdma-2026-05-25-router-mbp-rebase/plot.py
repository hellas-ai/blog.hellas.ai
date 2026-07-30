#!/usr/bin/env python3
"""Plots for the Mac ↔ Linux apple_rdma bench (2026-05-25 router-mbp-rebase).

Reads three CSVs produced by the apple_rdma bench harness:
  - uc_send.csv         single-QP UC SEND throughput by direction × depth × size
  - uc_lat.csv          ping-pong latency by direction × size
  - jaccl_allreduce.csv JACCL all-reduce bus bandwidth by layout × size

And writes three SVGs (same naming convention as bench/perftest/):
  - uc_send_bw.svg
  - uc_lat.svg
  - jaccl_allreduce_bw.svg
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter

BENCH_DIR = Path(__file__).parent

DIR_COLOR = {
    "linux-to-mac": "#1f77b4",
    "mac-to-linux": "#d62728",
}
DIR_LABEL = {
    "linux-to-mac": "Linux → Mac",
    "mac-to-linux": "Mac → Linux",
}
LAYOUT_COLOR = {
    "linux0-mac1": "#1f77b4",
    "mac0-linux1": "#d62728",
}
LAYOUT_LABEL = {
    "linux0-mac1": "Linux is rank 0 (Linux initiates)",
    "mac0-linux1": "Mac is rank 0 (Mac initiates)",
}
DEPTH_LINESTYLE = {1: ":", 8: "--", 32: "-"}


def human_size(n_bytes, _pos=None):
    n = int(n_bytes)
    if n >= 1 << 20:
        return f"{n // (1 << 20)} MiB"
    if n >= 1 << 10:
        return f"{n // (1 << 10)} KiB"
    return f"{n} B"


def style_size_axis(ax, sizes):
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(FuncFormatter(human_size))
    ax.set_xticks(sorted(set(sizes)))
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.tick_params(axis="x", labelsize=8)


def plot_uc_send():
    rows = []
    with (BENCH_DIR / "uc_send.csv").open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "direction": r["direction"],
                "size": int(r["size_bytes"]),
                "depth": int(r["depth"]),
                "gbps": float(r["rate_mbps"]) / 1000.0,
            })

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    groups = defaultdict(list)
    for r in rows:
        groups[(r["direction"], r["depth"])].append((r["size"], r["gbps"]))
    for (d, depth), pts in sorted(groups.items()):
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys,
                color=DIR_COLOR[d],
                linestyle=DEPTH_LINESTYLE.get(depth, "-"),
                marker="o", markersize=4, linewidth=1.6)

    style_size_axis(ax, [r["size"] for r in rows])
    ax.set_xlabel("Message size")
    ax.set_ylabel("Throughput (Gb/s)")
    ax.set_ylim(bottom=0)
    ax.axhline(40, color="gray", linestyle=":", alpha=0.6, linewidth=1)
    ax.annotate(
        "40 Gb/s — raw per-cable USB4 ceiling",
        xy=(0.98, 40), xycoords=("axes fraction", "data"),
        xytext=(0, -3), textcoords="offset points",
        fontsize=8, color="gray", ha="right", va="top",
    )
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title(
        "Single-QP UC SEND throughput, MacBook Pro ↔ router (AMD 8845HS)  "
        "(apple_rdma 2026-05-25)",
        fontsize=11,
    )
    dir_handles = [
        Line2D([0], [0], color=DIR_COLOR[d], marker="o", markersize=4,
               linewidth=2, label=DIR_LABEL[d])
        for d in ("linux-to-mac", "mac-to-linux")
    ]
    depth_handles = [
        Line2D([0], [0], color="black", linestyle=ls, linewidth=1.5,
               label=f"depth {d}")
        for d, ls in DEPTH_LINESTYLE.items()
    ]
    leg1 = ax.legend(handles=dir_handles, loc="upper left",
                     title="Direction", frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=depth_handles, loc="lower right",
              title="Inflight depth", frameon=True)

    fig.tight_layout()
    out = BENCH_DIR / "uc_send_bw.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_uc_lat():
    rows = []
    with (BENCH_DIR / "uc_lat.csv").open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "direction": r["direction"],
                "size": int(r["size_bytes"]),
                "p50": float(r["p50_us"]),
                "p99": float(r["p99_us"]),
                "min": float(r["min_us"]),
            })

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    by_dir = defaultdict(list)
    for r in rows:
        by_dir[r["direction"]].append(r)
    for d, rs in sorted(by_dir.items()):
        rs.sort(key=lambda x: x["size"])
        xs = [r["size"] for r in rs]
        ax.plot(xs, [r["p50"] for r in rs],
                color=DIR_COLOR[d], linestyle="-", marker="o",
                markersize=4, linewidth=1.6)
        ax.plot(xs, [r["p99"] for r in rs],
                color=DIR_COLOR[d], linestyle="--", marker="s",
                markersize=3, linewidth=1.2, alpha=0.7)

    style_size_axis(ax, [r["size"] for r in rows])
    ax.set_xlabel("Message size")
    ax.set_ylabel("Round-trip latency (µs)")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title(
        "Single-QP UC ping-pong latency, MacBook Pro ↔ router (AMD 8845HS)  "
        "(apple_rdma 2026-05-25)",
        fontsize=11,
    )
    dir_handles = [
        Line2D([0], [0], color=DIR_COLOR[d], marker="o", markersize=4,
               linewidth=2, label=DIR_LABEL[d])
        for d in ("linux-to-mac", "mac-to-linux")
    ]
    stat_handles = [
        Line2D([0], [0], color="black", linestyle="-", marker="o",
               markersize=4, linewidth=1.6, label="median"),
        Line2D([0], [0], color="black", linestyle="--", marker="s",
               markersize=3, linewidth=1.2, alpha=0.7, label="p99"),
    ]
    leg1 = ax.legend(handles=dir_handles, loc="upper left",
                     title="Direction", frameon=True)
    ax.add_artist(leg1)
    ax.legend(handles=stat_handles, loc="lower right",
              title="Statistic", frameon=True)

    fig.tight_layout()
    out = BENCH_DIR / "uc_lat.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_jaccl_allreduce():
    rows = []
    with (BENCH_DIR / "jaccl_allreduce.csv").open() as f:
        for r in csv.DictReader(f):
            if r["completed_ok"] != "1":
                continue
            rows.append({
                "layout": r["layout"],
                "size": int(r["size_bytes"]),
                # CSV column is bus_bw_gbps but the bench reports GB/s; convert.
                "gbps": float(r["bus_bw_gbps"]) * 8.0,
                "lat_us": float(r["latency_us"]),
            })

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    by_layout = defaultdict(list)
    for r in rows:
        by_layout[r["layout"]].append((r["size"], r["gbps"]))
    for layout, pts in sorted(by_layout.items()):
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(xs, ys,
                color=LAYOUT_COLOR[layout],
                marker="o", markersize=5, linewidth=1.8,
                label=LAYOUT_LABEL[layout])

    style_size_axis(ax, [r["size"] for r in rows])
    ax.set_xlabel("All-reduce payload size")
    ax.set_ylabel("Bus bandwidth (Gb/s)")
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title(
        "JACCL all-reduce, MacBook Pro M4 Max ↔ router (AMD 8845HS) over a single TB5 cable  "
        "(apple_rdma 2026-05-25)",
        fontsize=11,
    )
    ax.legend(loc="upper left", title="Layout", frameon=True)
    fig.tight_layout()
    out = BENCH_DIR / "jaccl_allreduce_bw.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def plot_mlx_lm_inference():
    """Distributed MLX-LM inference (Qwen2-0.5B-Instruct-4bit) across
    Mac + Linux ranks, sweeping prompt/gen token sizes. Each row pair
    is the same harness run reported from both ranks — average across
    ranks within a (layout, prompt_tokens) group."""
    rows = []
    with (BENCH_DIR / "mlx_lm_inference.csv").open() as f:
        for r in csv.DictReader(f):
            rows.append({
                "layout": r["layout"],
                "prompt_tokens": int(r["prompt_tokens"]),
                "generated_tokens": int(r["generated_tokens"]),
                "output_tok_s": float(r["output_tok_s"]),
            })

    by_key = defaultdict(list)
    for r in rows:
        by_key[(r["layout"], r["prompt_tokens"], r["generated_tokens"])
              ].append(r["output_tok_s"])

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    by_layout = defaultdict(list)
    for (layout, ptok, gtok), vals in by_key.items():
        avg = sum(vals) / len(vals)
        by_layout[layout].append((ptok, gtok, avg))
    for layout, pts in sorted(by_layout.items()):
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[2] for p in pts]
        ax.plot(xs, ys,
                color=LAYOUT_COLOR[layout],
                marker="o", markersize=5, linewidth=1.8,
                label=LAYOUT_LABEL[layout])

    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted({p[0] for layout in by_layout for p in by_layout[layout]}))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlabel("Prompt tokens (generated = prompt / 2 for each point)")
    ax.set_ylabel("Output throughput (tokens/s, averaged across ranks)")
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", alpha=0.25)
    ax.set_title(
        "MLX-LM distributed inference of Qwen2-0.5B-Instruct-4bit, "
        "MacBook Pro M4 Max ↔ router (AMD 8845HS)  "
        "(apple_rdma 2026-05-25)",
        fontsize=10,
    )
    ax.legend(loc="upper right", title="Layout", frameon=True)
    fig.tight_layout()
    out = BENCH_DIR / "mlx_lm_inference.svg"
    fig.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    plot_uc_send()
    plot_uc_lat()
    plot_jaccl_allreduce()
    plot_mlx_lm_inference()


if __name__ == "__main__":
    main()
