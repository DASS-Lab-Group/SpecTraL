#!/usr/bin/env python3
"""
apply_vitl_domainnet_patch.py
──────────────────────────────
Applies two targeted fixes to make ViT-L + DomainNet run without OOM:

  Fix 1 — DomainNet.py
      pin_memory=True  →  pin_memory=False
      Reason: with num_workers=0 (synchronous loading), pin_memory still
      allocates CUDA-pinned host buffers for EVERY DataLoader at construction
      time. With 100 clients that is 100 pinned buffers × 2 parallel
      experiments = 200 simultaneous pinned allocations, which exhausted
      the host pinned-memory pool and triggered the OOM crash.

  Fix 2 — main.py
      Add --num_workers CLI arg (default 2).
      Pass it through _build_domain_loaders → get_domainnet_dataset.
      Reason: num_workers was silently defaulting to 0 in the function
      signature, and was never reachable from the command line.

Run from your LoRA-FAIR repo root:
    python apply_vitl_domainnet_patch.py \
        --domainnet DomainNet.py \
        --main      main.py
"""

import argparse
import shutil
from pathlib import Path


def apply(filepath, replacements, label):
    p = Path(filepath)
    if not p.exists():
        print(f"  [SKIP] {filepath} not found.")
        return False
    text = p.read_text()
    backup = p.with_suffix(p.suffix + ".bak")
    shutil.copy(p, backup)
    applied = 0
    for old, new, desc in replacements:
        if old not in text:
            print(f"  [WARN] Pattern not found in {filepath}: {desc}")
            continue
        text = text.replace(old, new, 1)
        applied += 1
        print(f"  [OK]  {filepath} — {desc}")
    if applied:
        p.write_text(text)
        print(f"        Backup saved → {backup.name}\n")
    return applied > 0


# ══════════════════════════════════════════════════════════════════════════
# FIX 1: DomainNet.py — pin_memory=False
# ══════════════════════════════════════════════════════════════════════════
DOMAINNET_PATCHES = [
    (
        # OLD
        "clients_loader = [DataLoader(client, batch_size=batch_size, num_workers=num_workers, pin_memory=True,shuffle=True) \\\n"
        "                      for client in clients]\n"
        "    test_dloader = DataLoader(test_dataset, batch_size=256, num_workers=num_workers, pin_memory=True,\n"
        "                              shuffle=False)",
        # NEW
        "# pin_memory=False: even with num_workers=0, pin_memory=True allocates CUDA-pinned\n"
        "    # host buffers at DataLoader construction time. With 100 clients this exhausts the\n"
        "    # pinned-memory pool and causes OOM when running parallel experiments.\n"
        "    clients_loader = [DataLoader(client, batch_size=batch_size, num_workers=num_workers, pin_memory=False, shuffle=True) \\\n"
        "                      for client in clients]\n"
        "    test_dloader = DataLoader(test_dataset, batch_size=256, num_workers=num_workers, pin_memory=False,\n"
        "                              shuffle=False)",
        "pin_memory=True → False (train + test loaders)",
    ),
]


# ══════════════════════════════════════════════════════════════════════════
# FIX 2: main.py
# 2a — add --num_workers arg
# 2b — pass num_workers to get_domainnet_dataset
# ══════════════════════════════════════════════════════════════════════════
MAIN_PATCHES = [
    # 2a: add --num_workers after --batch_size
    (
        "    parser.add_argument(\"--batch_size\", type=int, default=32)",
        "    parser.add_argument(\"--batch_size\", type=int, default=32)\n"
        "    parser.add_argument(\"--num_workers\", type=int, default=2,\n"
        "                        help=\"DataLoader worker processes per domain. \"\n"
        "                             \"0=synchronous (safe but slow). \"\n"
        "                             \"2=recommended for ViT-L + DomainNet parallel runs.\")",
        "add --num_workers CLI arg",
    ),
    # 2b: pass num_workers to domainnet getter
    (
        "        if args.dataset == \"domainnet\":\n"
        "            tr, te = getter(\n"
        "                base_path=args.data_path,\n"
        "                domain_name=domain_name,\n"
        "                batch_size=args.batch_size,\n"
        "                alpha=args.alpha,\n"
        "                clients_for_eachdomain=clients_this_domain,\n"
        "                classnum=100,\n"
        "            )",
        "        if args.dataset == \"domainnet\":\n"
        "            tr, te = getter(\n"
        "                base_path=args.data_path,\n"
        "                domain_name=domain_name,\n"
        "                batch_size=args.batch_size,\n"
        "                alpha=args.alpha,\n"
        "                clients_for_eachdomain=clients_this_domain,\n"
        "                classnum=100,\n"
        "                num_workers=args.num_workers,\n"
        "            )",
        "pass num_workers to get_domainnet_dataset",
    ),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domainnet", default="DomainNet.py",
                    help="Path to DomainNet.py in your codebase")
    ap.add_argument("--main",      default="main.py",
                    help="Path to main.py in your codebase")
    args = ap.parse_args()

    print("=" * 60)
    print("Applying ViT-L / DomainNet OOM patches")
    print("=" * 60)

    print("\n[1] DomainNet.py — pin_memory fix")
    apply(args.domainnet, DOMAINNET_PATCHES, "DomainNet")

    print("[2] main.py — --num_workers arg + pass-through")
    apply(args.main, MAIN_PATCHES, "main")

    print("Done. Review backups (.bak) if anything looks wrong.")
    print("=" * 60)


if __name__ == "__main__":
    main()
