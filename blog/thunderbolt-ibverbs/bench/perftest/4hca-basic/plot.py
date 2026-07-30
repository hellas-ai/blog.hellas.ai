#!/usr/bin/env python3
"""Plot 4-HCA basic perftest results across two generations of runs.

Reads three CSVs from this directory:
  - 2026-05-25 single-rail kind={bw,bw_total,lat} sweep (write/read/send +
    64B latency on a single HCA). Used only for the latency chart now.
  - 2026-05-27 1 MiB aggregate run (mode={fwd,rev} × write/send × 4 HCAs,
    all four rails driven simultaneously, IOMMU off).
  - 2026-05-27 64 KiB aggregate run (same schema as above, smaller size).

Emits three SVGs:
  - bw_per_hca.svg     : 1 MiB aggregate per-HCA bandwidth, fwd vs rev,
                         with 4-rail totals annotated.
  - bw_per_hca_64k.svg : same chart at 64 KiB.
  - lat_us.svg         : 64 B single-HCA latency (write/read/send) from
                         the 2026-05-25 sweep.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt

BENCH_DIR = Path(__file__).parent

HCAS = ["usb4_rdma0", "usb4_rdma1", "usb4_rdma5", "usb4_rdma6"]
# (tool, direction) → (label, color, hatch)
SERIES = [
    ("ib_write_bw", "fwd", "ib_write_bw fwd",  "#1f77b4", ""),
    ("ib_write_bw", "rev", "ib_write_bw rev",  "#1f77b4", "//"),
    ("ib_send_bw",  "fwd", "ib_send_bw fwd",   "#ff7f0e", ""),
    ("ib_send_bw",  "rev", "ib_send_bw rev",   "#ff7f0e", "//"),
]
LAT_CASES = [
    ("write_lat_64", "ib_write_lat"),
    ("read_lat_64",  "ib_read_lat"),
    ("send_lat_64",  "ib_send_lat"),
]


def to_float(s) -> float | None:
    if s is None or s == "":
        return None
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def find_csv(suffix: str) -> Path | None:
    for p in sorted(BENCH_DIR.glob("*.csv")):
        if p.name.endswith(suffix):
            return p
    return None


def plot_agg(rows: list[dict], size_label: str, out_name: str) -> None:
    """Aggregate per-HCA bandwidth chart for fixed-size run."""
    # rows: mode/tool/dev → bw_avg_gbps
    table: dict[tuple[str, str, str], float] = {}
    for r in rows:
        if r.get("status") != "0":
            continue
        bw = to_float(r.get("bw_avg_gbps"))
        if bw is None:
            continue
        table[(r["tool"], r["mode"], r["dev"])] = bw

    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    n_series = len(SERIES)
    bar_w = 0.78 / n_series
    x = list(range(len(HCAS)))
    for i, (tool, mode, label, color, hatch) in enumerate(SERIES):
        offsets = [px + (i - (n_series - 1) / 2) * bar_w for px in x]
        ys = [table.get((tool, mode, hca), 0) for hca in HCAS]
        ax.bar(offsets, ys, bar_w, color=color,
               edgecolor="black", linewidth=0.4, label=label,
               hatch=hatch, alpha=0.92)

    # Per-(tool, mode) 4-rail totals as annotation text on the right.
    totals_text = ["4-rail aggregate totals:"]
    for tool, mode, label, _c, _h in SERIES:
        vals = [table.get((tool, mode, hca), 0) for hca in HCAS]
        totals_text.append(f"  {label}: {sum(vals):.1f} Gb/s")
    ax.text(
        1.005, 0.5, "\n".join(totals_text),
        transform=ax.transAxes, va="center", ha="left",
        fontsize=8, color="#222",
        bbox=dict(facecolor="white", edgecolor="#999", linewidth=0.5,
                  boxstyle="round,pad=0.4"),
    )

    ax.set_xticks(x)
    ax.set_xticklabels(HCAS, fontsize=9)
    ax.set_ylabel("Per-HCA average bandwidth (Gb/s)")
    ax.set_title(
        f"perftest 4-HCA aggregate — strix-1 ↔ strix-2 @ {size_label}  "
        "(IOMMU off, 2026-05-27)",
        fontsize=10,
    )
    ymax = max(table.values()) if table else 1.0
    ax.set_ylim(bottom=0, top=ymax * 1.25)
    ax.axhline(10, color="gray", linestyle=":", alpha=0.5, linewidth=1)
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8, frameon=True, ncol=2)
    fig.tight_layout(rect=(0, 0, 0.78, 1))
    fig.savefig(BENCH_DIR / out_name)
    plt.close(fig)
    print(f"Wrote {out_name}")


def plot_lat(rows: list[dict]) -> None:
    by_case: dict[str, float] = {}
    for r in rows:
        if r.get("kind") != "lat":
            continue
        v = to_float(r.get("lat_avg_us"))
        if v is None:
            continue
        by_case[r["case"]] = v

    fig, ax = plt.subplots(figsize=(8.5, 2.6))
    labels = [label for _, label in LAT_CASES]
    values = [by_case.get(c, 0) for c, _ in LAT_CASES]
    bars = ax.barh(labels, values,
                   color=["#1f77b4", "#ff7f0e", "#2ca02c"],
                   edgecolor="black", linewidth=0.5)
    vmax = max(values) if values else 0
    for bar, val in zip(bars, values):
        ax.text(val + vmax * 0.015,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f} µs",
                va="center", fontsize=9)
    ax.set_xlim(right=vmax * 1.18)
    ax.invert_yaxis()
    ax.set_xlabel("Average latency (µs, 64 B payload, single HCA)")
    ax.set_title(
        "perftest 4-HCA basic — 64B latency (usb4_rdma0, 2026-05-25)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(BENCH_DIR / "lat_us.svg")
    plt.close(fig)
    print("Wrote lat_us.svg")


def main() -> None:
    agg_1m = find_csv("1m-agg.csv")
    agg_64k = find_csv("64k-agg.csv")
    legacy = find_csv("perftest-4hca-basic.csv")

    if agg_1m:
        plot_agg(load_csv(agg_1m), "1 MiB", "bw_per_hca.svg")
    if agg_64k:
        plot_agg(load_csv(agg_64k), "64 KiB", "bw_per_hca_64k.svg")
    if legacy:
        plot_lat(load_csv(legacy))


if __name__ == "__main__":
    main()
