#!/usr/bin/env python3
"""Plot vLLM transport-matrix workload benchmarks.

Reads vllm-transport-matrix-*.csv files in this directory (May 2026 sweep
across strix-1 and strix-2, tp=2) and writes one SVG:
  - tps_vs_concurrency.svg : total throughput vs concurrency,
                              line per transport, single model headline.
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

BENCH_DIR = Path(__file__).parent

TRANSPORT_COLOR = {
    "solo": "#7f7f7f",
    "lan_tcp": "#2ca02c",
    "tb_tcp": "#d62728",
    "tb_rxe": "#9467bd",
    "usb4_rdma": "#1f77b4",
}
TRANSPORT_LABEL = {
    "solo": "solo (single host)",
    "lan_tcp": "TCP over 2.5G LAN",
    "tb_tcp": "TCP over TBnet",
    "tb_rxe": "RXE over TBnet",
    "usb4_rdma": "native usb4_rdma",
}
# Order legend by expected throughput (best → worst on networked)
TRANSPORT_ORDER = ["solo", "usb4_rdma", "tb_rxe", "tb_tcp", "lan_tcp"]

HEADLINE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct"
MODEL_SLUG = {
    "unsloth/Meta-Llama-3.1-8B-Instruct": "Llama-3.1-8B-Instruct",
    "Qwen/Qwen3-0.6B": "Qwen3-0.6B",
    "Qwen/Qwen3-4B-Instruct-2507": "Qwen3-4B-Instruct",
    "HuggingFaceTB/SmolLM3-3B": "SmolLM3-3B",
    "allenai/Olmo-3-7B-Instruct": "Olmo-3-7B-Instruct",
}


def load_rows():
    rows = []
    for csv_path in sorted(BENCH_DIR.glob("vllm-transport-matrix-*.csv")):
        with csv_path.open() as f:
            for r in csv.DictReader(f):
                if r.get("status") != "ok" or not r.get("total_tps"):
                    continue
                try:
                    rows.append({
                        "transport": r["transport"],
                        "model": r["model"],
                        "concurrency": int(r["concurrency"]),
                        "total_tps": float(r["total_tps"]),
                    })
                except (ValueError, KeyError):
                    pass
    return rows


def plot_tps_vs_concurrency(rows, model_key=HEADLINE_MODEL):
    matching = [r for r in rows if r["model"] == model_key]
    if not matching:
        print(f"no rows for {model_key}")
        return

    groups = defaultdict(list)
    for r in matching:
        groups[r["transport"]].append((r["concurrency"], r["total_tps"]))

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for t in TRANSPORT_ORDER:
        pts = groups.get(t)
        if not pts:
            continue
        pts.sort()
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys,
                color=TRANSPORT_COLOR.get(t, "gray"),
                marker="o", markersize=4, linewidth=1.6,
                label=TRANSPORT_LABEL.get(t, t))

    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 4, 16, 64, 256])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v)}"))
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.set_xlabel("Concurrency (in-flight requests)")
    ax.set_ylabel("End-to-end throughput (tokens/s)")
    short = MODEL_SLUG.get(model_key, model_key)
    ax.set_title(
        f"vLLM transport matrix: {short}, tp=2 strix-1 ↔ strix-2 (2026-05-02)"
    )
    ax.set_ylim(bottom=0)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="upper left", title="Transport", frameon=True)

    plt.tight_layout()
    out = BENCH_DIR / "tps_vs_concurrency.svg"
    plt.savefig(out)
    plt.close(fig)
    print(f"Wrote {out}")


def main():
    rows = load_rows()
    plot_tps_vs_concurrency(rows)


if __name__ == "__main__":
    main()
