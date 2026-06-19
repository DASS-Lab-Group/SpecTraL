#!/usr/bin/env python3
"""
Federated Vision LoRA Framework with multiple aggregation schemes.
"""

import argparse
import copy
import json
import logging
import math
import os
import sys
from pathlib import Path

import torch
import yaml

from client import Client
from server import Server
from utils import evaluation
from datasets.DomainNet import get_domainnet_dataset
from datasets.NICOPP import get_nico_dataset


HETER_RANK_LEVELS = [64, 32, 16, 8, 4]
HETER_RANK_PROFILES = {
    "heavy_tail_light": [0.10, 0.10, 0.20, 0.20, 0.40],
    "heavy_tail_strong": [0.40, 0.20, 0.20, 0.10, 0.10],
    "uniform": [0.20, 0.20, 0.20, 0.20, 0.20],
}


def setup_logging(log_level="INFO"):
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('lora_fair.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Federated Vision LoRA Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dataset", type=str, default="domainnet", choices=["domainnet", "nicopp"])
    parser.add_argument("--data_path", type=str, default="./datasets/DomainNet")
    parser.add_argument("--model", type=str, default="ViT", choices=["ViT", "ViT_L", "mixer"])
    parser.add_argument("--num_classes", type=int, default=345)

    parser.add_argument("--clients", type=int, default=30, help="Total number of clients")
    parser.add_argument("--client_fraction", type=float, default=0.6, help="Fraction of clients sampled each round")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local_epochs", type=int, default=3)
    parser.add_argument("--max_iterations", type=int, default=10000000)

    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument(
        "--aggregation",
        type=str,
        default="lora_fair",
        choices=["lora_fair", "fedit", "ffa", "flora", "flex", "florist", "spectral", "hetlora"],
    )

    parser.add_argument("--learning_rate", type=float, default=0.01) # for optmising local clients training
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--refinement_iterations", type=int, default=1000)
    parser.add_argument("--refinement_lr", type=float, default=0.01) # for optmising delta B on Server
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument(
        "--florist_rank_method",
        type=str,
        default="threshold",
        choices=["threshold", "gavish_donoho", "screenot"],
        help="Rank selection rule for FLoRIST.",
    )
    parser.add_argument("--florist_screenot_k", type=int, default=-1, help="Upper-bound k for ScreeNOT; -1 means auto.")
    parser.add_argument(
        "--florist_screenot_strategy",
        type=str,
        default="i",
        choices=["i", "w", "0"],
        help="ScreeNOT pseudo-noise strategy.",
    )
    parser.add_argument(
        "--florist_pad_init",
        type=str,
        default="zero",
        choices=["zero", "normal_a", "trained_a", "svd_w0_a", "svd_w0_a_vsqrt", "orthogonal_a"],
        help=(
            "FLoRIST train-init mode on clients: "
            "'zero' pads A/B with zeros; "
            "'normal_a' pads A with N(0,0.02) and B with zeros; "
            "'trained_a' sends server A at transmit max-rank while B keeps optimal-rank and zero-pads the rest; "
            "'svd_w0_a' initializes missing A rows as Sigma*V^T from SVD(W0); "
            "'svd_w0_a_vsqrt' initializes missing A rows as sqrt(Sigma)*V^T; "
            "'orthogonal_a' initializes missing A rows as random orthogonal-complement vectors; "
            "all SVD/orthogonal variants keep B zero-padded."
        ),
    )
    parser.add_argument("--florist_debug_svals", action="store_true", help="Print FLoRIST singular value diagnostics.")
    parser.add_argument("--florist_sv_topk", type=int, default=20, help="How many singular values to print per layer; <=0 prints all.")
    parser.add_argument("--florist_sv_eps", type=float, default=1e-8, help="Epsilon for counting non-zero singular values.")

    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet alpha for label non-IID inside each domain")

    parser.add_argument("--heter", action="store_true", help="Enable heterogeneous client ranks")
    parser.add_argument(
        "--heter_rank_profile",
        type=str,
        default="heavy_tail_light",
        choices=["heavy_tail_light", "heavy_tail_strong", "uniform"],
        help=(
            "Rank profile used when --heter is enabled and --local_ranks is not provided. "
            "heavy_tail_light: more low-rank clients, heavy_tail_strong: more high-rank clients, "
            "uniform: equal fraction across rank levels."
        ),
    )
    parser.add_argument(
        "--local_ranks",
        type=str,
        default="",
        help=(
            "Comma-separated ranks for heter mode; if provided, overrides --heter_rank_profile "
            "and is cycled/truncated to #clients."
        ),
    )

    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])

    parser.add_argument("--output_dir", type=str, default="./results")
    parser.add_argument("--save_model", action="store_true")
    parser.add_argument(
        "--ablation",
        action="store_true",
        help=(
            "After round 1, compute singular values of stacked LoRA weights (spectral/QR method) "
            "and all four threshold cut points (energy@0.9, energy@0.99, Gavish-Donoho, ScreeNOT) "
            "per layer. Results are saved to a JSON file in --output_dir."
        ),
    )
    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--ablation_every", type=int, default=5,
                        help="Save ablation JSON every N rounds (and always on the final round).")
    parser.add_argument(
        "--deltaw_sanity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Compute round-wise sanity error between expected weighted avg delta-W and eval global adapter delta-W.",
    )
    parser.add_argument(
        "--deltaw_sanity_topk",
        type=int,
        default=5,
        help="Number of worst LoRA layers to print in delta-W sanity check.",
    )

    return parser.parse_args()


def load_config(config_path):
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    return {}


def setup_environment(args):
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        # Cap cuDNN workspace to 256 MB per process to avoid OOM when running
        # multiple ViT-L jobs simultaneously.  cuDNN is kept enabled for speed.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True  # TF32 for matmuls (attention/MLP in ViT)
        torch.cuda.set_per_process_memory_fraction(0.90)
    Path(_build_run_dir(args)).mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"
        logging.warning("CUDA not available, using CPU")


def _parse_local_ranks(s, n_clients):
    if s is None or str(s).strip() == "":
        return []
    vals = [int(x.strip()) for x in s.split(',') if x.strip()]
    if not vals:
        return []
    if len(vals) >= n_clients:
        return vals[:n_clients]
    out = []
    i = 0
    while len(out) < n_clients:
        out.append(vals[i % len(vals)])
        i += 1
    return out


def _allocate_counts_from_fractions(n_clients, fractions):
    if not fractions:
        raise ValueError("fractions must be non-empty")
    if any(float(f) < 0 for f in fractions):
        raise ValueError("fractions must be non-negative")
    s = float(sum(float(f) for f in fractions))
    if s <= 0:
        raise ValueError("fractions must sum to a positive value")
    norm = [float(f) / s for f in fractions]
    raw = [p * float(n_clients) for p in norm]
    counts = [int(math.floor(x)) for x in raw]
    rem = int(n_clients - sum(counts))
    if rem > 0:
        frac_parts = [(raw[i] - counts[i], i) for i in range(len(raw))]
        frac_parts.sort(key=lambda t: t[0], reverse=True)
        for _, idx in frac_parts[:rem]:
            counts[idx] += 1
    return counts


def _build_profile_ranks(n_clients, profile_name):
    if profile_name not in HETER_RANK_PROFILES:
        raise ValueError(
            f"Invalid heter_rank_profile '{profile_name}'. "
            f"Choose from {sorted(HETER_RANK_PROFILES.keys())}"
        )
    counts = _allocate_counts_from_fractions(n_clients, HETER_RANK_PROFILES[profile_name])
    ranks = []
    for rank, cnt in zip(HETER_RANK_LEVELS, counts):
        ranks.extend([int(rank)] * int(cnt))
    if len(ranks) != n_clients:
        raise RuntimeError(
            f"Internal profile rank allocation error: expected {n_clients}, got {len(ranks)}"
        )
    return ranks


def _build_domain_loaders(args, logger):
    if args.dataset == "domainnet":
        domain_names = ['clipart', 'infograph', 'painting', 'quickdraw', 'real', 'sketch']
        getter = get_domainnet_dataset
    elif args.dataset == "nicopp":
        domain_names = ['autumn', 'dim', 'grass', 'outdoor', 'rock', 'water']
        getter = get_nico_dataset
    else:
        raise NotImplementedError(f"Dataset {args.dataset} not implemented")

    base = args.clients // len(domain_names)
    rem = args.clients % len(domain_names)

    train_loaders, test_loaders, eval_domains = [], [], []

    for i, domain_name in enumerate(domain_names):
        clients_this_domain = base + (1 if i < rem else 0)
        if clients_this_domain == 0:
            continue

        if args.dataset == "domainnet":
            tr, te = getter(
                base_path=args.data_path,
                domain_name=domain_name,
                batch_size=args.batch_size,
                alpha=args.alpha,
                clients_for_eachdomain=clients_this_domain,
                classnum=100,
                num_workers=args.num_workers,
            )
        else:
            tr, te = getter(
                base_path=args.data_path,
                domain_name=domain_name,
                batch_size=args.batch_size,
                alpha=args.alpha,
                clients_for_eachdomain=clients_this_domain,
                num_workers=args.num_workers,
            )

        train_loaders.extend(tr)
        test_loaders.append(te)
        eval_domains.append(domain_name)

    logger.info(
        f"Data split: feature non-IID by domain + label non-IID Dirichlet(alpha={args.alpha}); "
        f"total_clients={len(train_loaders)}, eval_domains={eval_domains}"
    )

    return train_loaders, test_loaders, eval_domains


def create_clients(args, train_loaders):
    n_clients = len(train_loaders)
    if args.heter:
        local_ranks = _parse_local_ranks(args.local_ranks, n_clients)
        if local_ranks:
            client_ranks = local_ranks
        else:
            client_ranks = _build_profile_ranks(n_clients, args.heter_rank_profile)
    else:
        client_ranks = [args.lora_rank] * n_clients
    clients = []
    for i, dataloader in enumerate(train_loaders):
        client = Client(
            dataloader=dataloader,
            num_layers=24 if args.model == 'ViT_L' else 12,
            num_classes=args.num_classes,
            depth_cls=0,
            modeltype=args.model,
            rank=client_ranks[i],
        )
        clients.append(client)
    return clients, client_ranks


def create_server(args):
    return Server(
        num_layers=24 if args.model == 'ViT_L' else 12,
        num_classes=args.num_classes,
        depth_cls=0,
        modeltype=args.model,
        domains_num=args.clients,
        clients_per_domain=1,
        rank=args.lora_rank,
    )


def _aggregate_by_method(args, server, client_parameters, selected_ranks, all_client_ranks, client_weights):
    if args.aggregation == "lora_fair":
        return server.aggregate_lora_fair(
            client_parameters,
            args.refinement_iterations,
            args.refinement_lr,
            args.lambda_reg,
            client_ranks=selected_ranks,
            client_weights=client_weights,
        )
    if args.aggregation == "fedit":
        return server.aggregate_fedit(client_parameters, client_ranks=selected_ranks, client_weights=client_weights)
    if args.aggregation == "hetlora":
        return server.aggregate_hetlora(client_parameters, client_ranks=selected_ranks, client_weights=client_weights)
    if args.aggregation == "ffa":
        return server.aggregate_ffa(client_parameters, client_ranks=selected_ranks, client_weights=client_weights)
    if args.aggregation == "flora":
        return server.aggregate_flora(client_parameters, client_ranks=selected_ranks, client_weights=client_weights)
    if args.aggregation == "flex":
        return server.aggregate_flex(
            client_parameters,
            client_ranks=selected_ranks,
            global_max_rank=max(all_client_ranks),
            client_weights=client_weights,
        )
    if args.aggregation == "florist":
        return server.aggregate_florist(
            client_parameters,
            threshold=args.threshold,
            rank_method=args.florist_rank_method,
            screenot_k=args.florist_screenot_k,
            screenot_strategy=args.florist_screenot_strategy,
            client_ranks=selected_ranks,
            global_max_rank=max(all_client_ranks),
            train_init_mode=args.florist_pad_init,
            client_weights=client_weights,
            debug_svals=args.florist_debug_svals,
            sv_topk=args.florist_sv_topk,
            sv_eps=args.florist_sv_eps,
        )
    if args.aggregation == "spectral":
        return server.aggregate_spectral(
            client_parameters,
            threshold=args.threshold,
            rank_method=args.florist_rank_method,
            screenot_k=args.florist_screenot_k,
            screenot_strategy=args.florist_screenot_strategy,
            client_ranks=selected_ranks,
            global_max_rank=max(all_client_ranks),
            train_init_mode=args.florist_pad_init,
            round_idx=None,
            client_weights=client_weights,
            debug_svals=args.florist_debug_svals,
            sv_topk=args.florist_sv_topk,
            sv_eps=args.florist_sv_eps,
        )
    raise ValueError(f"Unsupported aggregation method: {args.aggregation}")


def _sample_clients(n_clients, frac, seed):
    n_sel = max(1, int(round(frac * n_clients)))
    n_sel = min(n_clients, n_sel)
    g = torch.Generator()
    g.manual_seed(seed)
    idx = torch.randperm(n_clients, generator=g)[:n_sel].tolist()
    idx.sort()
    return idx


def _client_data_size(client):
    if hasattr(client, "dataloader") and hasattr(client.dataloader, "dataset"):
        try:
            return int(len(client.dataloader.dataset))
        except Exception:
            pass
    if hasattr(client, "dataloader"):
        try:
            return int(len(client.dataloader))
        except Exception:
            pass
    return 1


def _sanitize_tag(v):
    return str(v).replace(" ", "_").replace("/", "_").replace(".", "p")


def _reshape_lora_a_for_product(a):
    if not torch.is_tensor(a):
        return None
    if a.ndim == 2:
        return a
    if a.ndim >= 3:
        return a.reshape(a.shape[0], -1)
    return None


def _reshape_lora_b_for_product(b):
    if not torch.is_tensor(b):
        return None
    if b.ndim == 2:
        return b
    if b.ndim >= 3:
        return b.reshape(b.shape[0], -1)
    return None


def _normalize_weights(weights, n):
    if weights is None:
        return [1.0 / float(n)] * n
    w = [float(x) for x in weights]
    if len(w) != n:
        raise ValueError(f"Expected {n} weights, got {len(w)}")
    s = sum(w)
    if s <= 0:
        return [1.0 / float(n)] * n
    return [x / s for x in w]


def _compute_expected_deltaw(client_parameters, client_weights):
    weights = _normalize_weights(client_weights, len(client_parameters))
    expected = {}
    for cp, w in zip(client_parameters, weights):
        for key, a in cp.items():
            if "lora_A" not in key:
                continue
            b_key = key.replace("lora_A", "lora_B")
            if b_key not in cp:
                continue
            a2 = _reshape_lora_a_for_product(a.detach().to(dtype=torch.float32, device="cpu"))
            b2 = _reshape_lora_b_for_product(cp[b_key].detach().to(dtype=torch.float32, device="cpu"))
            if a2 is None or b2 is None or b2.shape[1] != a2.shape[0]:
                continue
            prod = torch.matmul(b2, a2)
            if key not in expected:
                expected[key] = prod * w
            else:
                expected[key] = expected[key] + prod * w
    return expected


def _compute_global_deltaw(global_state):
    out = {}
    for key, a in global_state.items():
        if "lora_A" not in key:
            continue
        b_key = key.replace("lora_A", "lora_B")
        if b_key not in global_state:
            continue
        a2 = _reshape_lora_a_for_product(a.detach().to(dtype=torch.float32, device="cpu"))
        b2 = _reshape_lora_b_for_product(global_state[b_key].detach().to(dtype=torch.float32, device="cpu"))
        if a2 is None or b2 is None or b2.shape[1] != a2.shape[0]:
            continue
        out[key] = torch.matmul(b2, a2)
    return out


def _summarize_deltaw_sanity(expected_deltaw, global_deltaw, topk=5):
    expected_keys = set(expected_deltaw.keys())
    global_keys = set(global_deltaw.keys())
    common_keys = sorted(expected_keys.intersection(global_keys))

    total_abs_sq = 0.0
    total_exp_sq = 0.0
    total_global_sq = 0.0
    per_layer = []
    skipped_shape = 0

    for key in common_keys:
        exp_dw = expected_deltaw[key]
        glob_dw = global_deltaw[key]
        if exp_dw.shape != glob_dw.shape:
            skipped_shape += 1
            continue

        diff = glob_dw - exp_dw
        abs_sq = float(torch.sum(diff * diff).item())
        exp_sq = float(torch.sum(exp_dw * exp_dw).item())
        glob_sq = float(torch.sum(glob_dw * glob_dw).item())
        abs_fro = math.sqrt(max(abs_sq, 0.0))
        exp_fro = math.sqrt(max(exp_sq, 0.0))
        glob_fro = math.sqrt(max(glob_sq, 0.0))
        rel_fro = abs_fro / (exp_fro + 1e-12)

        total_abs_sq += abs_sq
        total_exp_sq += exp_sq
        total_global_sq += glob_sq
        per_layer.append(
            {
                "key": key,
                "abs_fro": abs_fro,
                "expected_fro": exp_fro,
                "global_fro": glob_fro,
                "rel_fro": rel_fro,
            }
        )

    per_layer_sorted = sorted(per_layer, key=lambda x: x["rel_fro"], reverse=True)
    agg_abs = math.sqrt(max(total_abs_sq, 0.0))
    agg_exp = math.sqrt(max(total_exp_sq, 0.0))
    agg_global = math.sqrt(max(total_global_sq, 0.0))
    agg_rel = agg_abs / (agg_exp + 1e-12)
    mean_rel = float(sum(x["rel_fro"] for x in per_layer) / len(per_layer)) if per_layer else None
    max_rel = float(per_layer_sorted[0]["rel_fro"]) if per_layer_sorted else None

    return {
        "layers_compared": int(len(per_layer)),
        "layers_common": int(len(common_keys)),
        "layers_expected_only": int(len(expected_keys - global_keys)),
        "layers_global_only": int(len(global_keys - expected_keys)),
        "layers_skipped_shape_mismatch": int(skipped_shape),
        "agg_abs_fro": float(agg_abs),
        "agg_expected_fro": float(agg_exp),
        "agg_global_fro": float(agg_global),
        "agg_rel_fro": float(agg_rel),
        "mean_layer_rel_fro": mean_rel,
        "max_layer_rel_fro": max_rel,
        "top_layers": per_layer_sorted[: max(0, int(topk))],
    }


def _florist_method_tag(args):
    if args.florist_rank_method == "gavish_donoho":
        return "gavish_donoho"
    elif args.florist_rank_method == "screenot":
        k_tag = "auto" if args.florist_screenot_k < 0 else str(args.florist_screenot_k)
        return f"screenot_{args.florist_screenot_strategy}_k{k_tag}"
    else:
        return f"threshold_{_sanitize_tag(args.threshold)}"


def _build_run_dir(args):
    """Return a run-specific subdirectory under output_dir.

    Structure:
        {output_dir}/{model}/{dataset}/{aggregation}/
            {mode}_r{lora_rank}_rounds{rounds}_c{clients}_f{frac}_ep{epochs}_lr{lr}_a{alpha}_s{seed}[_{method}[_pad_{pad}]]/
    """
    mode_tag = "heter" if args.heter else "homo"
    run_parts = [
        mode_tag,
        f"r{args.lora_rank}",
        f"rounds{args.rounds}",
        f"c{args.clients}",
        f"f{_sanitize_tag(args.client_fraction)}",
        f"ep{args.local_epochs}",
        f"lr{_sanitize_tag(args.learning_rate)}",
        f"a{_sanitize_tag(args.alpha)}",
        f"s{args.seed}",
    ]
    if args.aggregation in ("florist", "spectral"):
        run_parts.append(_florist_method_tag(args))
        run_parts.append(f"pad_{args.florist_pad_init}")
    return os.path.join(
        args.output_dir,
        _sanitize_tag(args.model),
        _sanitize_tag(args.dataset),
        _sanitize_tag(args.aggregation),
        "_".join(run_parts),
    )


def _build_rank_file(args):
    return os.path.join(_build_run_dir(args), "optimal_ranks.jsonl")


def _build_ablation_file(args, round_num):
    return os.path.join(_build_run_dir(args), f"ablation_round{round_num + 1}.json")


def _build_results_file(args):
    return os.path.join(_build_run_dir(args), "results.json")


def run_federated_learning(args, clients, client_ranks, server, test_loaders, eval_domains, logger):
    florist_rank_file = None
    results_file = _build_results_file(args)
    if args.aggregation in ("florist", "spectral"):
        florist_rank_file = _build_rank_file(args)
        logger.info(f"{args.aggregation} rank records will be saved to: {florist_rank_file}")
    logger.info(f"Results file: {results_file}")

    results = {
        'round': [],
        'avg_top1': [],
        'avg_top5': [],
        'top1_per_domain': [],
        'top5_per_domain': [],
        'round_losses': [], #
        'eval_domains': eval_domains,
        'selected_clients_per_round': [],
        'round_rank_comm': [],
        'round_param_comm': [],
        'total_rank_comm': 0,
        'total_param_comm': 0,
        'deltaw_sanity': [],
    }
    if florist_rank_file is not None:
        results['florist_rank_file'] = florist_rank_file
    results['results_file'] = results_file

    logger.info(f"Starting federated learning with total_clients={len(clients)}, rounds={args.rounds}, participation_fraction={args.client_fraction}")

    for round_num in range(args.rounds):
        logger.info(f"Round {round_num + 1}/{args.rounds}")
        logger.info(f"[MEM] Start of round: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")

        selected = _sample_clients(len(clients), args.client_fraction, args.seed + round_num)
        selected_ranks = [client_ranks[i] for i in selected]
        selected_sizes = [_client_data_size(clients[i]) for i in selected]
        total_size = float(sum(selected_sizes)) if sum(selected_sizes) > 0 else float(len(selected_sizes))
        client_weights = [float(s) / total_size for s in selected_sizes]
        results['selected_clients_per_round'].append(selected)
        logger.info(f"Selected clients ({len(selected)}): {selected}")
        logger.info(f"Selected client sizes: {selected_sizes}")
        logger.info(f"Aggregation weights: {[round(w, 6) for w in client_weights]}")

        client_parameters = []
        client_losses = [] #
        for pos, i in enumerate(selected):
            logger.info(f"Training client {i} ({pos + 1}/{len(selected)}) rank={client_ranks[i]}")
            logger.info(f"[MEM] Before training client {i}: {torch.cuda.memory_allocated()/1e9:.2f} GB")
            if round_num > 0:
                if args.aggregation in ("florist", "spectral"):
                    global_params = server.get_florist_train_parameters()
                else:
                    global_params = server.get_full_parameters()
                if args.aggregation == "flora":
                    clients[i].load_and_prepare_flora_parameters(global_params)
                elif args.aggregation in ("florist", "spectral"):
                    clients[i].load_parameters(
                        global_params,
                        florist_pad_mode=args.florist_pad_init,
                    )
                else:
                    clients[i].load_parameters(global_params)
            else:
                logger.info(f"Round 1 init: client {i} uses local initialization (no global load).")
            client_loss =clients[i].train_baseline(args.learning_rate, args.local_epochs, args.max_iterations, method=args.aggregation) # capturing the loss as train_baseline returns client_loss
            client_parameters.append(clients[i].get_full_parameters())
            client_losses.append(client_loss) #
            logger.info(f"[MEM] After training client {i}: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        logger.info(f"[MEM] Before aggregation: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        logger.info(f"Performing aggregation method={args.aggregation}")
        rank_comm, param_comm = _aggregate_by_method(
            args,
            server,
            client_parameters,
            selected_ranks,
            client_ranks,
            client_weights,
        )
        rank_comm = int(rank_comm if rank_comm is not None else 0)
        param_comm = int(param_comm if param_comm is not None else 0)
        results['round_rank_comm'].append(rank_comm)
        results['round_param_comm'].append(param_comm)
        results['total_rank_comm'] += rank_comm
        results['total_param_comm'] += param_comm
        logger.info(
            f"Aggregation complete | round_rank={rank_comm}, round_params={param_comm}, "
            f"total_rank={results['total_rank_comm']}, total_params={results['total_param_comm']}"
        )
        logger.info(f"[MEM] After aggregation: {torch.cuda.memory_allocated()/1e9:.2f} GB allocated")
        if args.aggregation in ("florist", "spectral"):
            rank_record = server.get_latest_florist_rank_record()
            if rank_record is not None:
                with open(florist_rank_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rank_record) + "\n")
                logger.info(f"Saved {args.aggregation} optimal ranks for round {round_num + 1}")

        # Ablation study: compute singular values and all four cuts per layer.
        _save_ablation = (
            args.ablation
            and (
                round_num == 0
                or (round_num + 1) % args.ablation_every == 0
                or round_num == args.rounds - 1
            )
        )
        if _save_ablation:
            ablation_file = _build_ablation_file(args, round_num)
            logger.info(f"[Ablation] Computing singular values and threshold cuts (spectral/QR method)...")
            ablation_data = server.compute_ablation_svals(
                client_parameters,
                client_weights=client_weights,
                screenot_k=args.florist_screenot_k,
                screenot_strategy=args.florist_screenot_strategy,
            )
            ablation_meta = {
                "dataset": args.dataset,
                "model": args.model,
                "lora_rank": args.lora_rank,
                "clients": args.clients,
                "client_fraction": args.client_fraction,
                "heter": args.heter,
                "seed": args.seed,
                "screenot_strategy": args.florist_screenot_strategy,
                "screenot_k": args.florist_screenot_k,
                "round": round_num + 1,
                "layers": ablation_data,
            }
            with open(ablation_file, "w", encoding="utf-8") as f:
                json.dump(ablation_meta, f, indent=2)
            logger.info(f"[Ablation] Saved singular values and cuts for {len(ablation_data)} layers to: {ablation_file}")
        avg_loss = sum(client_losses) / len(client_losses) #
        results['round'].append(round_num + 1)
        results['round_losses'].append(avg_loss) #
        logger.info(f"Round {round_num + 1} - Average Loss: {avg_loss:.4f}") #
        # Save results after every round
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            logger.info(f"Results saved to {results_file}")

        if ((round_num + 1) % args.eval_every == 0) or (round_num == args.rounds - 1):
            logger.info(f"Evaluating at round {round_num + 1}")
            # Compute before freeing client_parameters — only needed on eval rounds ~ Tanishk
            if ((round_num + 1) % args.eval_every == 0) or (round_num == args.rounds - 1):
                expected_deltaw = _compute_expected_deltaw(client_parameters, client_weights) if args.deltaw_sanity else None
            else:
                expected_deltaw = None
            del client_parameters
            torch.cuda.empty_cache()
            # expected_deltaw = _compute_expected_deltaw(client_parameters, client_weights) if args.deltaw_sanity else None
            sanity_global_state = None
            sanity_state_name = None
            if args.aggregation == "flora":
                eval_model = copy.deepcopy(server.global_model)
                if args.deltaw_sanity:
                    # Sanity must use the exact model loaded for eval, but before merge.
                    sanity_global_state = copy.deepcopy(eval_model.state_dict())
                    sanity_state_name = "copy(server.global_model) pre-merge state_dict()"
                eval_model.backbone = eval_model.backbone.merge_and_unload()
                logger.info("Evaluation model: merged stacked global adapter (FLoRA)")
            elif args.aggregation in ("florist", "spectral"):
                eval_model = server.get_eval_model_for_florist()
                if args.deltaw_sanity:
                    sanity_global_state = eval_model.state_dict()
                    sanity_state_name = "server.get_eval_model_for_florist().state_dict()"
                logger.info(
                    f"Evaluation model: raw post-threshold {args.aggregation} global adapter"
                )
            elif args.aggregation == "flex":
                eval_ranks = sorted(set(client_ranks), reverse=True)
                logger.info(f"Evaluation model: FlexLoRA truncated globals over ranks={eval_ranks}")
                rank_data_sizes = {}
                for ci, r in enumerate(client_ranks):
                    rank_data_sizes[r] = rank_data_sizes.get(r, 0) + _client_data_size(clients[ci])
                total_rank_data = sum(rank_data_sizes.values())
                if total_rank_data > 0:
                    rank_weights = {r: rank_data_sizes[r] / total_rank_data for r in eval_ranks}
                else:
                    rank_weights = {r: 1.0 / len(eval_ranks) for r in eval_ranks}
                logger.info(f"Flex eval rank weights: { {r: round(w, 6) for r, w in rank_weights.items()} }")

                top1_acc = None
                top5_acc = None
                flex_sanity_records = []
                for r in eval_ranks:
                    eval_model = server.get_eval_model_for_rank(r)
                    w = rank_weights[r]
                    if args.deltaw_sanity:
                        global_deltaw_r = _compute_global_deltaw(eval_model.state_dict())
                        sanity_r = _summarize_deltaw_sanity(
                            expected_deltaw,
                            global_deltaw_r,
                            topk=args.deltaw_sanity_topk,
                        )
                        sanity_r["rank"] = int(r)
                        sanity_r["weight"] = float(w)
                        flex_sanity_records.append(sanity_r)
                    t1_r, t5_r = evaluation(eval_model, test_loaders)
                    if not isinstance(t1_r, list):
                        t1_r, t5_r = [t1_r], [t5_r]
                    if top1_acc is None:
                        top1_acc = [0.0] * len(t1_r)
                        top5_acc = [0.0] * len(t5_r)
                    top1_acc = [acc + w * val for acc, val in zip(top1_acc, t1_r)]
                    top5_acc = [acc + w * val for acc, val in zip(top5_acc, t5_r)]
                    avg1_r = sum(t1_r) / len(t1_r)
                    avg5_r = sum(t5_r) / len(t5_r)
                    logger.info(
                        f"Flex eval rank={r} | weight={w:.4f} | "
                        f"Avg Top-1={avg1_r:.2f}% Avg Top-5={avg5_r:.2f}%"
                    )
                if args.deltaw_sanity:
                    weighted_abs = float(sum(x["agg_abs_fro"] * x["weight"] for x in flex_sanity_records))
                    weighted_rel = float(sum(x["agg_rel_fro"] * x["weight"] for x in flex_sanity_records))
                    sanity = {
                        "round": int(round_num + 1),
                        "aggregation": args.aggregation,
                        "state_source": "server.get_eval_model_for_rank(r).state_dict() (weighted over eval ranks)",
                        "weighted_abs_fro": weighted_abs,
                        "weighted_rel_fro": weighted_rel,
                        "by_rank": flex_sanity_records,
                    }
                    results["deltaw_sanity"].append(sanity)
                    logger.info(
                        f"DeltaW sanity (Flex weighted over eval ranks) | abs_fro={weighted_abs:.6e} | rel_fro={weighted_rel:.6e}"
                    )
            else:
                eval_model = server.global_model
                if args.deltaw_sanity:
                    sanity_global_state = eval_model.state_dict()
                    sanity_state_name = "server.global_model.state_dict()"
                logger.info("Evaluation model: server.global_model")

            if args.deltaw_sanity and args.aggregation != "flex":
                global_deltaw = _compute_global_deltaw(sanity_global_state)
                sanity = _summarize_deltaw_sanity(
                    expected_deltaw,
                    global_deltaw,
                    topk=args.deltaw_sanity_topk,
                )
                sanity['round'] = int(round_num + 1)
                sanity['aggregation'] = args.aggregation
                sanity['state_source'] = sanity_state_name
                results['deltaw_sanity'].append(sanity)
                logger.info(
                    f"DeltaW sanity ({sanity_state_name}) | layers={sanity['layers_compared']} "
                    f"| abs_fro={sanity['agg_abs_fro']:.6e} | rel_fro={sanity['agg_rel_fro']:.6e} "
                    f"| expected_fro={sanity['agg_expected_fro']:.6e} | global_fro={sanity['agg_global_fro']:.6e}"
                )
                if sanity['layers_skipped_shape_mismatch'] > 0:
                    logger.warning(
                        f"DeltaW sanity: skipped {sanity['layers_skipped_shape_mismatch']} layer(s) due to shape mismatch"
                    )
                for idx, entry in enumerate(sanity['top_layers'], start=1):
                    logger.info(
                        f"DeltaW sanity worst[{idx}] key={entry['key']} | rel={entry['rel_fro']:.6e} "
                        f"| abs={entry['abs_fro']:.6e} | expected={entry['expected_fro']:.6e} "
                        f"| global={entry['global_fro']:.6e}"
                    )
               # After the deltaw sanity logging block ends:
            if expected_deltaw is not None:
                del expected_deltaw
                expected_deltaw = None

            if args.aggregation != "flex":
                top1_acc, top5_acc = evaluation(eval_model, test_loaders)
                if eval_model is not server.global_model: # ~tanishk
                    del eval_model
                    torch.cuda.empty_cache()

            if not isinstance(top1_acc, list):
                top1_acc, top5_acc = [top1_acc], [top5_acc]

            avg_top1 = sum(top1_acc) / len(top1_acc)
            avg_top5 = sum(top5_acc) / len(top5_acc)



            logger.info(f"Round {round_num + 1} - Per-domain Top-1: {dict(zip(eval_domains, [round(x, 2) for x in               top1_acc]))}")
            logger.info(f"Round {round_num + 1} - Average Top-1: {avg_top1:.2f}%")
            logger.info(f"Round {round_num + 1} - Average Top-5: {avg_top5:.2f}%")

            # results['round'].append(round_num + 1) commenting as it's already being called in every loop
            results['avg_top1'].append(avg_top1)
            results['avg_top5'].append(avg_top5)
            results['top1_per_domain'].append(top1_acc)
            results['top5_per_domain'].append(top5_acc)

    return results


def save_results(args, results, logger):
    results_file = results.get('results_file', _build_results_file(args))
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {results_file}")

    if results['avg_top1']:
        logger.info(
            f"Final Results - Top-1: {results['avg_top1'][-1]:.2f}%, "
            f"Top-5: {results['avg_top5'][-1]:.2f}%, "
            f"Total rank comm: {results['total_rank_comm']}, Total param comm: {results['total_param_comm']}"
            f"Total Average Loss: {results['round_losses']}"
        )


def main():
    args = parse_arguments()
    logger = setup_logging(args.log_level)
    logger.info("Federated LoRA Fine-Tuning")
    logger.info(f"Configuration: {vars(args)}")

    if not os.path.exists(args.data_path):
        logger.error(f"Data path does not exist: {args.data_path}")
        logger.error("Please update --data_path to point to your dataset directory")
        sys.exit(1)

    setup_environment(args)
    logger.info(f"Loading {args.dataset} dataset from {args.data_path}")

    train_loaders, test_loaders, eval_domains = _build_domain_loaders(args, logger)
    clients, client_ranks = create_clients(args, train_loaders)
    if args.heter:
        rank_dist = {}
        for r in client_ranks:
            rank_dist[int(r)] = rank_dist.get(int(r), 0) + 1
        logger.info(f"Heter rank distribution: {rank_dist}")
    else:
        logger.info(f"Homogeneous rank: {args.lora_rank}")
    server = create_server(args)

    logger.info(f"Created {len(clients)} clients and 1 server")

    results = run_federated_learning(args, clients, client_ranks, server, test_loaders, eval_domains, logger)
    save_results(args, results, logger)

    if args.save_model:
        model_path = os.path.join(_build_run_dir(args), "model.pth")
        torch.save(server.global_model.state_dict(), model_path)
        logger.info(f"Model saved to {model_path}")

    logger.info("Experiment completed successfully!")


if __name__ == "__main__":
    main()
