#!/usr/bin/env python3
"""
Plot per-round FLoRIST thresholded optimal ranks.

Input format: JSONL file produced by main.py.
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="Plot FLoRIST optimal ranks per round")
    p.add_argument(
        "--rank_file",
        type=str,
        required=True,
        help="Path to florist_optimal_ranks.jsonl",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory to save round plots (default: <rank_file_dir>/florist_rank_plots)",
    )
    return p.parse_args()


def load_records(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def make_round_plot(record, out_path):
    details = record.get("rank_details", [])
    grouped = defaultdict(list)
    fallback_x = 0

    for d in details:
        matrix = d.get("matrix", d.get("key", "unknown"))
        layer = d.get("layer")
        if layer is None:
            layer = fallback_x
            fallback_x += 1
        grouped[matrix].append((int(layer), int(d.get("optimal_rank", 0))))

    plt.figure(figsize=(10, 6))
    for matrix, points in sorted(grouped.items()):
        points = sorted(points, key=lambda x: x[0])
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        plt.plot(xs, ys, marker="o", linewidth=1.5, label=matrix)

    round_id = record.get("round", "?")
    thr = record.get("threshold", "?")
    rank_method = record.get("rank_method", "threshold")
    if rank_method == "gavish_donoho":
        plt.title(f"FLoRIST Optimal Ranks | Round {round_id} | method=gavish_donoho")
    elif rank_method == "screenot":
        strat = record.get("screenot_strategy", "i")
        k = record.get("screenot_k", "auto")
        plt.title(f"FLoRIST Optimal Ranks | Round {round_id} | method=screenot ({strat}, k={k})")
    else:
        plt.title(f"FLoRIST Optimal Ranks | Round {round_id} | method=threshold ({thr})")
    plt.xlabel("Layer")
    plt.ylabel("Optimal Rank")
    plt.grid(True, linestyle="--", alpha=0.4)
    if grouped:
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    args = parse_args()
    records = load_records(args.rank_file)
    if not records:
        raise ValueError(f"No records found in {args.rank_file}")

    stem = os.path.splitext(os.path.basename(args.rank_file))[0]
    out_dir = args.out_dir or os.path.join(os.path.dirname(args.rank_file), f"florist_rank_plots_{stem}")
    os.makedirs(out_dir, exist_ok=True)

    for idx, rec in enumerate(records, start=1):
        round_id = rec.get("round", idx)
        out_path = os.path.join(out_dir, f"{stem}_round_{int(round_id):03d}.png")
        make_round_plot(rec, out_path)

    print(f"Saved {len(records)} round plots to: {out_dir}")


if __name__ == "__main__":
    main()
