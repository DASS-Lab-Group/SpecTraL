#!/usr/bin/env python3
"""
memory_monitor.py — Brev A100 memory monitor for LoRA-FAIR experiments.

Monitors: GPU VRAM (per GPU), CPU RAM, /dev/shm (shared memory),
Swap, and per-process breakdown. Logs to file and prints with colour.

Usage:
    # Run in background while experiments are running:
    nohup python memory_monitor.py --interval 10 --log mem.log &

    # Single snapshot:
    python memory_monitor.py --once

    # Custom thresholds:
    python memory_monitor.py --gpu_warn 70 --gpu_crit 88 --cpu_crit 85 --shm_crit 70

    # Show more processes:
    python memory_monitor.py --top_n 15
"""

import argparse
import os
import re
import sys
import shutil
import subprocess
import time
import datetime
from pathlib import Path

try:
    import psutil
except ImportError:
    print("[monitor] Installing psutil...")
    subprocess.check_call([sys.executable, "-m", "pip", "install",
                           "psutil", "--break-system-packages", "-q"])
    import psutil

# ── ANSI colours ──────────────────────────────────────────────────────────
RED    = "\033[91m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"
DIM    = "\033[2m"

def _col(pct, warn, crit):
    if pct >= crit:   return RED
    if pct >= warn:   return YELLOW
    return GREEN

def _bar(pct, width=28):
    n = int(width * min(pct, 100) / 100)
    return "[" + "█" * n + "░" * (width - n) + "]"

def _gb(b):
    return b / (1024 ** 3)

def _strip_ansi(s):
    return re.sub(r'\033\[[0-9;]*m', '', s)


# ── GPU ───────────────────────────────────────────────────────────────────
def _gpu_stats():
    """Per-GPU memory and utilisation via nvidia-smi."""
    try:
        raw = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return []
    gpus = []
    for line in raw.split("\n"):
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        idx, name, used_mb, free_mb, total_mb, util = parts
        used, free, total = float(used_mb), float(free_mb), float(total_mb)
        gpus.append(dict(
            idx=int(idx), name=name,
            used_gb=used/1024, free_gb=free/1024, total_gb=total/1024,
            pct=100*used/total if total else 0,
            util=float(util),
        ))
    return gpus


def _gpu_procs():
    """GPU memory per process via nvidia-smi compute-apps."""
    try:
        # uuid → index mapping
        uid_raw = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,gpu_uuid", "--format=csv,noheader"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
        uid2idx = {}
        for ln in uid_raw.split("\n"):
            pts = [p.strip() for p in ln.split(",")]
            if len(pts) == 2:
                uid2idx[pts[1]] = int(pts[0])
    except Exception:
        uid2idx = {}

    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,used_memory",
             "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return []

    procs = []
    for line in raw.split("\n"):
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        try:
            pid, uuid, mem_mb = int(parts[0]), parts[1], float(parts[2])
            try:
                name = psutil.Process(pid).name()
                cmd  = " ".join(psutil.Process(pid).cmdline())[:70]
            except Exception:
                name, cmd = "?", ""
            procs.append(dict(pid=pid, gpu=uid2idx.get(uuid,-1),
                               mem_gb=mem_mb/1024, name=name, cmd=cmd))
        except Exception:
            continue
    procs.sort(key=lambda x: x["mem_gb"], reverse=True)
    return procs


# ── System ────────────────────────────────────────────────────────────────
def _sys_stats():
    vm = psutil.virtual_memory()
    sw = psutil.swap_memory()
    shm_used = shm_total = shm_free = 0
    if Path("/dev/shm").exists():
        s = shutil.disk_usage("/dev/shm")
        shm_total, shm_used, shm_free = s.total, s.used, s.free
    return dict(
        ram_pct   = vm.percent,
        ram_used  = _gb(vm.used),
        ram_total = _gb(vm.total),
        ram_avail = _gb(vm.available),
        swap_pct  = sw.percent,
        swap_used = _gb(sw.used),
        swap_total= _gb(sw.total),
        shm_pct   = 100*shm_used/shm_total if shm_total else 0,
        shm_used  = _gb(shm_used),
        shm_total = _gb(shm_total),
        shm_free  = _gb(shm_free),
    )


def _top_procs(n=10):
    procs = []
    for p in psutil.process_iter(['pid','name','memory_info','cpu_percent','cmdline']):
        try:
            rss = _gb(p.info['memory_info'].rss)
            cmd = " ".join(p.info.get('cmdline') or [])[:65] or p.info['name']
            procs.append(dict(pid=p.info['pid'], name=p.info['name'],
                               rss=rss, cpu=p.info['cpu_percent'], cmd=cmd))
        except Exception:
            pass
    procs.sort(key=lambda x: x['rss'], reverse=True)
    return procs[:n]


# ── Render ─────────────────────────────────────────────────────────────────
def render(args, logfile=None):
    ts   = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    SEP  = "═" * 74
    sep2 = "─" * 74
    lines = []

    def L(s=""): lines.append(s)

    L(f"\n{BOLD}{CYAN}{SEP}{RESET}")
    L(f"{BOLD}   Brev Memory Monitor        {ts}{RESET}")
    L(f"{BOLD}{CYAN}{SEP}{RESET}")

    alerts = []

    # ── GPU VRAM ──────────────────────────────────────────────────────────
    L(f"\n{BOLD}[ GPU VRAM ]{RESET}")
    gpus = _gpu_stats()
    if not gpus:
        L("  nvidia-smi unavailable.")
    else:
        for g in gpus:
            c = _col(g['pct'], args.gpu_warn, args.gpu_crit)
            L(f"  GPU{g['idx']}  {g['name'][:30]:<30}  "
              f"{c}{_bar(g['pct'])} {g['pct']:5.1f}%{RESET}  "
              f"{g['used_gb']:.2f}/{g['total_gb']:.0f} GiB  "
              f"util:{g['util']:.0f}%")
            if g['pct'] >= args.gpu_crit:
                alerts.append(f"GPU {g['idx']} VRAM {g['pct']:.1f}% "
                               f"({g['used_gb']:.2f}/{g['total_gb']:.0f} GiB)")

    # GPU processes
    gprocs = _gpu_procs()
    if gprocs:
        L(f"\n  {BOLD}Processes on GPU:{RESET}")
        L(f"  {'PID':>7}  {'GPU':>3}  {'GiB':>6}  {'Name':<14}  Command")
        L(f"  {DIM}{'─'*65}{RESET}")
        for gp in gprocs:
            c = RED if gp['mem_gb'] > 10 else (YELLOW if gp['mem_gb'] > 5 else RESET)
            L(f"  {gp['pid']:>7}  GPU{gp['gpu']:>1}  "
              f"{c}{gp['mem_gb']:>5.2f}{RESET}  {gp['name']:<14}  {gp['cmd'][:50]}")

    # ── CPU RAM ───────────────────────────────────────────────────────────
    s = _sys_stats()
    L(f"\n{BOLD}[ CPU RAM ]{RESET}")
    c = _col(s['ram_pct'], 70, args.cpu_crit)
    L(f"  RAM   {c}{_bar(s['ram_pct'])} {s['ram_pct']:5.1f}%{RESET}  "
      f"{s['ram_used']:.1f}/{s['ram_total']:.0f} GiB  "
      f"avail: {s['ram_avail']:.1f} GiB")
    if s['ram_pct'] >= args.cpu_crit:
        alerts.append(f"CPU RAM {s['ram_pct']:.1f}% "
                       f"({s['ram_used']:.1f}/{s['ram_total']:.0f} GiB)")

    # ── Shared Memory ─────────────────────────────────────────────────────
    L(f"\n{BOLD}[ Shared Memory  /dev/shm ]{RESET}")
    if s['shm_total'] > 0:
        c = _col(s['shm_pct'], 50, args.shm_crit)
        L(f"  SHM   {c}{_bar(s['shm_pct'])} {s['shm_pct']:5.1f}%{RESET}  "
          f"{s['shm_used']:.2f}/{s['shm_total']:.0f} GiB  "
          f"free: {s['shm_free']:.2f} GiB")
        if s['shm_pct'] >= args.shm_crit:
            alerts.append(f"/dev/shm {s['shm_pct']:.1f}% "
                           f"({s['shm_used']:.2f}/{s['shm_total']:.0f} GiB)")
    else:
        L("  /dev/shm not mounted.")

    # ── Swap ──────────────────────────────────────────────────────────────
    L(f"\n{BOLD}[ Swap ]{RESET}")
    if s['swap_total'] > 0:
        c = _col(s['swap_pct'], 15, 40)
        L(f"  Swap  {c}{_bar(s['swap_pct'])} {s['swap_pct']:5.1f}%{RESET}  "
          f"{s['swap_used']:.1f}/{s['swap_total']:.0f} GiB")
        if s['swap_pct'] >= 15:
            alerts.append(f"Swap active: {s['swap_used']:.1f} GiB — RAM pressure")
    else:
        L("  No swap configured.")

    # ── Top Processes by RAM ──────────────────────────────────────────────
    L(f"\n{BOLD}[ Top {args.top_n} Processes by CPU RAM ]{RESET}")
    L(f"  {'PID':>7}  {'Name':<16}  {'RSS GiB':>8}  {'CPU%':>6}  Command")
    L(f"  {DIM}{sep2}{RESET}")
    for p in _top_procs(args.top_n):
        c = RED if p['rss'] > 20 else (YELLOW if p['rss'] > 8 else GREEN)
        L(f"  {p['pid']:>7}  {p['name']:<16}  "
          f"{c}{p['rss']:>7.2f}{RESET}  {p['cpu']:>5.1f}%  {p['cmd'][:55]}")

    # ── Alert Summary ─────────────────────────────────────────────────────
    if alerts:
        L(f"\n{BOLD}{RED}[ ⚠  ALERTS  — {len(alerts)} threshold(s) exceeded ]{RESET}")
        for a in alerts:
            L(f"  {RED}▶  {a}{RESET}")
    else:
        L(f"\n{GREEN}  ✓  All memory within thresholds.{RESET}")

    L(f"{CYAN}{SEP}{RESET}\n")

    out = "\n".join(lines)
    print(out)

    if logfile:
        with open(logfile, "a") as f:
            f.write(_strip_ansi(out))

    return bool(alerts)


# ── Entry ─────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Brev A100 memory monitor")
    ap.add_argument("--interval",  type=int,   default=10,
                    help="Poll interval in seconds (default 10)")
    ap.add_argument("--gpu_warn",  type=float, default=70.0,
                    help="GPU VRAM warning threshold %%")
    ap.add_argument("--gpu_crit",  type=float, default=88.0,
                    help="GPU VRAM critical alert threshold %%")
    ap.add_argument("--cpu_crit",  type=float, default=85.0,
                    help="CPU RAM critical alert threshold %%")
    ap.add_argument("--shm_crit",  type=float, default=70.0,
                    help="/dev/shm critical alert threshold %%")
    ap.add_argument("--top_n",     type=int,   default=10,
                    help="Number of top CPU-RAM processes to display")
    ap.add_argument("--log",       type=str,   default="memory_monitor.log",
                    help="Log file path (ANSI-stripped)")
    ap.add_argument("--once",      action="store_true",
                    help="Print one snapshot and exit")
    args = ap.parse_args()

    if args.once:
        render(args, logfile=args.log)
        return

    print(f"{BOLD}{CYAN}Memory monitor running — interval={args.interval}s  "
          f"log={args.log}{RESET}")
    print(f"Thresholds: GPU warn={args.gpu_warn}% crit={args.gpu_crit}%  "
          f"CPU={args.cpu_crit}%  SHM={args.shm_crit}%")
    print("Ctrl-C to stop.\n")
    try:
        while True:
            render(args, logfile=args.log)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Monitor stopped.{RESET}")

if __name__ == "__main__":
    main()
