"""
Server implementation with multiple aggregation schemes and heterogeneous-rank support.
"""

import copy
import re
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from peft import LoraConfig
from utils import FoundationModel


class Server:
    def __init__(self, num_layers=12, num_classes=10, depth_cls=0, modeltype='ViT', domains_num=6, clients_per_domain=1, rank=8):
        self.modeltype = modeltype
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.depth_cls = depth_cls
        self.client_num = domains_num * clients_per_domain
        self.rank = rank

        self.global_model = self._build_lora_model(rank)
        self.delta_lora_B = {}
        self.florist_eval_state = None
        self.florist_eval_rank = None
        self.florist_train_state = None
        self.latest_florist_rank_record = None
        # Store LoRA-free base model state for FLoRA lifecycle.
        base_model = self._build_base_model()
        # FIX (move to CPU — freed from GPU immediately): ~tanishk
        self.global_base_state = {
        k: v.cpu() for k, v in base_model.backbone.state_dict().items()
        }     
        del base_model
        torch.cuda.empty_cache()
        

    def _ensure_global_rank(self, target_rank):
        target_rank = int(target_rank)
        if target_rank <= 0:
            target_rank = self.rank
        if target_rank == self.rank:
            return
        self.rank = target_rank
        self.global_model = self._build_lora_model(self.rank)

    def _build_lora_model(self, rank):
        if self.modeltype in ('ViT', 'ViT_L'):
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=max(8, 2 * rank),
                target_modules=['attn.proj', 'mlp.fc2'],
                lora_dropout=0.1,
                bias="none",
            )
        elif self.modeltype == 'mixer':
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=max(8, 2 * rank),
                target_modules=['mlp_tokens.fc2', 'mlp_channels.fc2'],
                lora_dropout=0.1,
                bias="none",
            )
        else:
            raise ValueError(f"Unsupported model type: {self.modeltype}")
        return FoundationModel(self.num_layers, self.num_classes, self.depth_cls, self.modeltype, lora_config).cuda()

    def _build_base_model(self):
        return FoundationModel(self.num_layers, self.num_classes, self.depth_cls, self.modeltype, lora_config=None).cuda()

    @staticmethod
    def _reshape_lora_a_for_svd(a):
        if a.ndim == 2:
            return a
        if a.ndim >= 3:
            return a.reshape(a.shape[0], -1)
        return None

    @staticmethod
    def _reshape_lora_b_for_svd(b):
        if b.ndim == 2:
            return b
        if b.ndim >= 3:
            return b.reshape(b.shape[0], -1)
        return None

    @staticmethod
    def _restore_lora_a_shape(a_2d, ref_a):
        if ref_a.ndim == 2:
            return a_2d
        return a_2d.reshape((a_2d.shape[0],) + tuple(ref_a.shape[1:]))

    @staticmethod
    def _restore_lora_b_shape(b_2d, ref_b):
        if ref_b.ndim == 2:
            return b_2d
        return b_2d.reshape((ref_b.shape[0], b_2d.shape[1]) + tuple(ref_b.shape[2:]))

    def _homogenize_state_to_rank(self, state, target_rank):
        out = {}
        for k, v in state.items():
            if 'lora_A' in k and v.ndim >= 2:
                r = v.shape[0]
                if r < target_rank:
                    pad_shape = (target_rank - r,) + tuple(v.shape[1:])
                    pad = torch.zeros(pad_shape, dtype=v.dtype, device=v.device)
                    out[k] = torch.cat([v, pad], dim=0)
                else:
                    out[k] = v[:target_rank, ...]
            elif 'lora_B' in k and v.ndim >= 2:
                r = v.shape[1]
                if r < target_rank:
                    pad_shape = (v.shape[0], target_rank - r) + tuple(v.shape[2:])
                    pad = torch.zeros(pad_shape, dtype=v.dtype, device=v.device)
                    out[k] = torch.cat([v, pad], dim=1)
                else:
                    out[k] = v[:, :target_rank, ...]
            else:
                out[k] = v
        return out

    @staticmethod
    def _normalize_weights(client_parameters, client_weights=None):
        num_clients = len(client_parameters)
        if client_weights is None:
            return [1.0 / num_clients] * num_clients
        weights = [float(w) for w in client_weights]
        if len(weights) != num_clients:
            raise ValueError(f"client_weights length {len(weights)} != num_clients {num_clients}")
        s = sum(weights)
        if s <= 0:
            return [1.0 / num_clients] * num_clients
        return [w / s for w in weights]

    @staticmethod
    def _average_states(client_parameters, client_weights=None):
        weights = Server._normalize_weights(client_parameters, client_weights)
        out = {}
        for k in client_parameters[0].keys():
            out[k] = sum(cp[k] * w for cp, w in zip(client_parameters, weights))
        return out

    @staticmethod
    def _compute_comm_stats(state_dict):
        total_rank = 0
        total_param = 0
        for k, v in state_dict.items():
            if not torch.is_tensor(v) or v.ndim < 2:
                continue
            if 'lora_A' in k:
                total_rank += v.shape[0]
                total_param += v.shape[0] * v.shape[1]
            elif 'lora_B' in k:
                total_rank += v.shape[1]
                total_param += v.shape[0] * v.shape[1]
        return int(total_rank), int(total_param)

    @staticmethod
    def _max_lora_rank_in_state(state_dict):
        max_rank = 0
        for k, v in state_dict.items():
            if not torch.is_tensor(v) or v.ndim < 2:
                continue
            if 'lora_A' in k:
                max_rank = max(max_rank, int(v.shape[0]))
            elif 'lora_B' in k:
                max_rank = max(max_rank, int(v.shape[1]))
        return max_rank

    def aggregate_fedit(self, client_parameters, client_ranks=None, client_weights=None):
        target_rank = max(client_ranks) if client_ranks else self.rank
        self._ensure_global_rank(target_rank)
        # Homogenize all client states to max selected client rank before averaging.
        homog = [self._homogenize_state_to_rank(cp, target_rank) for cp in client_parameters]
        global_params = self._average_states(homog, client_weights=client_weights)
        self.global_model.load_state_dict(global_params, strict=False)
        return self._compute_comm_stats(global_params)

    def aggregate_hetlora(self, client_parameters, client_ranks=None, client_weights=None):
        """HetLoRA: sparsity-weighted aggregation (Cho et al., EMNLP 2024).

        Instead of data-size weights, each client's contribution to a given LoRA
        layer is proportional to the Frobenius norm of its local delta-W = B @ A
        for that layer.  This down-weights clients whose updates are sparse/small
        (e.g. a high-rank client that didn't need the extra capacity) and
        up-weights clients with informative, high-magnitude updates.

        Distribution to clients uses the same truncation as fedit: each client
        receives the first r_k rows/cols of the global A/B.
        """
        target_rank = max(client_ranks) if client_ranks else self.rank
        self._ensure_global_rank(target_rank)

        # Non-LoRA params (head, prompt) use ordinary data-size weights.
        data_weights = self._normalize_weights(client_parameters, client_weights)
        global_params = copy.deepcopy(client_parameters[0])
        for k in list(global_params.keys()):
            if 'lora_' in k:
                continue
            global_params[k] = sum(cp[k] * w for cp, w in zip(client_parameters, data_weights))

        # Zero-pad every client adapter to target_rank once; reuse below.
        homog = [self._homogenize_state_to_rank(cp, target_rank) for cp in client_parameters]

        # Per-layer sparsity-weighted average of A and B.
        for key in (k for k in client_parameters[0].keys() if 'lora_A' in k):
            b_key = key.replace('lora_A', 'lora_B')

            # Compute ||B_k @ A_k||_F for each client using the original (un-padded) tensors.
            norms = []
            for cp in client_parameters:
                if key not in cp or b_key not in cp:
                    norms.append(0.0)
                    continue
                a2 = self._reshape_lora_a_for_svd(cp[key])
                b2 = self._reshape_lora_b_for_svd(cp[b_key])
                if a2 is None or b2 is None or b2.shape[1] != a2.shape[0]:
                    norms.append(0.0)
                    continue
                norms.append(float(torch.norm(b2 @ a2, p='fro').item()))

            total_norm = sum(norms)
            layer_weights = (
                [n / total_norm for n in norms]
                if total_norm > 0
                else [1.0 / len(client_parameters)] * len(client_parameters)
            )

            global_params[key] = sum(h[key] * w for h, w in zip(homog, layer_weights))
            global_params[b_key] = sum(h[b_key] * w for h, w in zip(homog, layer_weights))

        self.global_model.load_state_dict(global_params, strict=False)
        return self._compute_comm_stats(global_params)

    def aggregate_ffa(self, client_parameters, client_ranks=None, client_weights=None):
        return self.aggregate_fedit(client_parameters, client_ranks=client_ranks, client_weights=client_weights)

    def aggregate_flex(self, client_parameters, client_ranks=None, global_max_rank=None, client_weights=None):
        target_rank = int(global_max_rank) if global_max_rank is not None else (max(client_ranks) if client_ranks else self.rank)
        self._ensure_global_rank(target_rank)
        weights = self._normalize_weights(client_parameters, client_weights)
        global_params = copy.deepcopy(client_parameters[0])
        for k in list(global_params.keys()):
            if 'lora_' in k:
                continue
            global_params[k] = sum(cp[k] * w for cp, w in zip(client_parameters, weights))

        aggregated_ba = {}
        refs = {}
        for cp, w in zip(client_parameters, weights):
            for key, a in cp.items():
                if 'lora_A' not in key:
                    continue
                b_key = key.replace('lora_A', 'lora_B')
                if b_key not in cp:
                    continue
                a2 = self._reshape_lora_a_for_svd(a)
                b2 = self._reshape_lora_b_for_svd(cp[b_key])
                if a2 is None or b2 is None or b2.shape[1] != a2.shape[0]:
                    continue
                aggregated_ba[key] = aggregated_ba.get(key, 0) + (b2.cuda() @ a2.cuda()) * w
                if key not in refs:
                    refs[key] = (a, cp[b_key])

        for key, ba in aggregated_ba.items():
            b_key = key.replace('lora_A', 'lora_B')
            a_ref, b_ref = refs[key]
            u, s, vt = torch.linalg.svd(ba, full_matrices=False)
            k = min(target_rank, s.numel())
            global_params[key] = self._restore_lora_a_shape(vt[:k, :].cpu(), a_ref)
            global_params[b_key] = self._restore_lora_b_shape((u[:, :k] @ torch.diag(s[:k])).cpu(), b_ref)

        self.global_model.load_state_dict(global_params, strict=False)
        return self._compute_comm_stats(global_params)

    def aggregate_flora(self, client_parameters, client_ranks=None, client_weights=None):
        # Server job in FLoRA: weighted stacking of client LoRA adapters.
        if client_ranks is None:
            client_ranks = [self.rank] * len(client_parameters)

        weights = self._normalize_weights(client_parameters, client_weights)
        keys = [k for k in client_parameters[0].keys() if 'lora_A' in k]
        # In FLoRA, clients merge the previous global stacked adapter into base
        # before local training, so client non-LoRA params represent the new base.
        # Use client base (they should be identical across selected clients) to avoid
        # stale-base + new-LoRA mismatch.
        global_params = copy.deepcopy(client_parameters[0])
        for k in list(global_params.keys()):
            if 'lora_' in k:
                continue
            # Keep merged base consistent; only aggregate trainable non-LoRA params.
            # In this codebase, trainable non-LoRA params are head / Prompt.
            # if ('head' in k) or ('Prompt' in k):
            global_params[k] = sum(cp[k] * w for cp, w in zip(client_parameters, weights))
            # else:
            #     global_params[k] = client_parameters[0][k]

        stacked_state = {}
        for key in keys:
            b_key = key.replace('lora_A', 'lora_B')
            a_chunks = []
            b_chunks = []
            for cp, w in zip(client_parameters, weights):
                if b_key not in cp:
                    continue
                a_chunks.append(cp[key] * w)
                b_chunks.append(cp[b_key])
            if a_chunks:
                stacked_state[key] = torch.cat(a_chunks, dim=0)
                stacked_state[b_key] = torch.cat(b_chunks, dim=1)
                global_params[key] = stacked_state[key]
                global_params[b_key] = stacked_state[b_key]

        stacked_rank = int(sum(client_ranks)) if len(client_ranks) > 0 else self.rank
        self._ensure_global_rank(stacked_rank)
        self.global_model.load_state_dict(global_params, strict=False)
        return self._compute_comm_stats(stacked_state)

    @staticmethod
    def _extract_layer_and_matrix_name(key):
        layer = None
        m = re.search(r'blocks\.(\d+)\.', key)
        if m:
            layer = int(m.group(1))
        if layer is None:
            m2 = re.search(r'layers\.(\d+)\.', key)
            if m2:
                layer = int(m2.group(1))

        matrix = key
        m3 = re.search(r'blocks\.\d+\.(.+?)\.lora_A', key)
        if m3:
            matrix = m3.group(1)
        else:
            m4 = re.search(r'layers\.\d+\.(.+?)\.lora_A', key)
            if m4:
                matrix = m4.group(1)
        return layer, matrix

    @staticmethod
    def _omega_gavish_donoho(beta):
        # Polynomial approximation from Gavish-Donoho (unknown noise level), beta in [0, 1].
        return 0.56 * (beta ** 3) - 0.95 * (beta ** 2) + 1.82 * beta + 1.43

    @staticmethod
    def _select_rank_florist(
        s_vals,
        p_shape,
        threshold=0.9,
        rank_method="threshold",
        screenot_k=-1,
        screenot_strategy="i",
    ):
        if s_vals.numel() == 0:
            return 1, {"method": rank_method}

        if rank_method == "threshold":
            energy = torch.cumsum(s_vals ** 2, dim=0) / torch.sum(s_vals ** 2)
            idx = (energy >= threshold).nonzero(as_tuple=False)
            k_opt = (idx.min().item() + 1) if idx.numel() else s_vals.shape[0]
            return max(1, int(k_opt)), {"method": "threshold", "threshold": float(threshold)}

        if rank_method == "gavish_donoho":
            m, n = int(p_shape[0]), int(p_shape[1])
            mn = min(m, n)
            mx = max(m, n)
            beta = float(mn) / float(mx) if mx > 0 else 1.0
            omega = Server._omega_gavish_donoho(beta)
            sigma_med = torch.median(s_vals).item()
            tau = omega * sigma_med
            k_opt = int((s_vals > tau).sum().item())
            return max(1, k_opt), {
                "method": "gavish_donoho",
                "beta": float(beta),
                "omega": float(omega),
                "tau": float(tau),
                "median_sigma": float(sigma_med),
            }

        if rank_method == "screenot":
            try:
                from screenot.ScreeNOT import createPseudoNoise, computeOptThreshold
            except Exception as e:
                raise ImportError(
                    "ScreeNOT method selected but package 'screenot' is not available. "
                    "Install with: python3 -m pip install --user screenot"
                ) from e

            # Fast ScreeNOT: use the already-computed singular values directly.
            # This avoids an extra SVD inside adaptiveHardThresholding().
            fY = s_vals.detach().cpu().numpy()
            mn = min(int(p_shape[0]), int(p_shape[1]))
            if screenot_k is None or int(screenot_k) < 0:
                # Auto upper-bound: quarter of dimension, respecting ScreeNOT constraints.
                k_auto = max(1, mn // 4)
            else:
                k_auto = int(screenot_k)
            if screenot_strategy == "i":
                k_max = max(1, (mn - 2) // 2)
            else:
                k_max = max(1, mn - 1)
            k_used = max(1, min(k_auto, k_max))

            fZ = createPseudoNoise(fY, k=k_used, strategy=screenot_strategy)
            gamma = float(min(int(p_shape[0]) / int(p_shape[1]), int(p_shape[1]) / int(p_shape[0])))
            tau = float(computeOptThreshold(fZ, gamma))
            r = int(np.sum(fY > tau))
            k_opt = max(1, int(r))
            return k_opt, {
                "method": "screenot",
                "strategy": screenot_strategy,
                "k_used": int(k_used),
                "tau": float(tau),
                "impl": "fast_svals",
            }

        raise ValueError(f"Unknown rank_method: {rank_method}")

    def aggregate_florist(
        self,
        client_parameters,
        threshold=0.9,
        rank_method="threshold",
        screenot_k=-1,
        screenot_strategy="i",
        client_ranks=None,
        global_max_rank=None,
        train_init_mode="zero",
        round_idx=None,
        client_weights=None,
        debug_svals=False,
        sv_topk=20,
        sv_eps=1e-8,
        aggregation_name="florist",
        stacked_factorization="svd",
    ):
        if client_ranks is None:
            client_ranks = [self.rank] * len(client_parameters)

        if train_init_mode not in ("zero", "normal_a", "trained_a", "svd_w0_a", "svd_w0_a_vsqrt", "orthogonal_a"):
            raise ValueError(f"Unsupported FLoRIST train_init_mode: {train_init_mode}")
        if stacked_factorization not in ("svd", "qr"):
            raise ValueError(f"Unsupported stacked_factorization: {stacked_factorization}")
        log_prefix = "[FLoRIST]" if aggregation_name == "florist" else "[Spectral]"

        weights = self._normalize_weights(client_parameters, client_weights)
        rank_details = []
        train_transmit_rank = (
            int(global_max_rank)
            if global_max_rank is not None
            else (max(client_ranks) if client_ranks else self.rank)
        )
        train_transmit_rank = max(1, int(train_transmit_rank))
        if train_init_mode == "trained_a":
            print(
                f"{log_prefix} train_init_mode=trained_a | transmit_rank={train_transmit_rank} "
                f"(A sent at transmit rank; B optimal+zero-pad)"
            )
        elif train_init_mode == "svd_w0_a":
            print(
                f"{log_prefix} train_init_mode=svd_w0_a | clients will init padded A rows from SVD(W0); "
                "server keeps strict optimal-rank adapters."
            )
        elif train_init_mode == "svd_w0_a_vsqrt":
            print(
                f"{log_prefix} train_init_mode=svd_w0_a_vsqrt | clients will init padded A rows with sqrt(Sigma)*V^T from SVD(W0); "
                "server keeps strict optimal-rank adapters."
            )
        elif train_init_mode == "orthogonal_a":
            print(
                f"{log_prefix} train_init_mode=orthogonal_a | clients will init padded A rows with orthogonal-complement vectors; "
                "server keeps strict optimal-rank adapters."
            )

        # Do not homogenize before stacked factorization; aggregate raw client adapters first.
        global_params_eval = copy.deepcopy(client_parameters[0])
        for k in list(global_params_eval.keys()):
            if 'lora_' in k:
                continue
            global_params_eval[k] = sum(cp[k] * w for cp, w in zip(client_parameters, weights))
        global_params_train = copy.deepcopy(global_params_eval)

        keys = [k for k in client_parameters[0].keys() if 'lora_A' in k]
        for key in keys:
            b_key = key.replace('lora_A', 'lora_B')
            a_chunks = []
            b_chunks = []
            a_ref = None
            b_ref = None
            for cp, w in zip(client_parameters, weights):
                if b_key not in cp:
                    continue
                a2 = self._reshape_lora_a_for_svd(cp[key])
                b2 = self._reshape_lora_b_for_svd(cp[b_key])
                if a2 is None or b2 is None or b2.shape[1] != a2.shape[0]:
                    continue
                a_chunks.append(a2 * w)
                b_chunks.append(b2)
                if a_ref is None:
                    a_ref = cp[key]
                    b_ref = cp[b_key]
            if not a_chunks:
                continue

            a = torch.cat(a_chunks, dim=0)
            b = torch.cat(b_chunks, dim=1)
            a_gpu = a.cuda()
            b_gpu = b.cuda()

            if stacked_factorization == "svd":
                u_b, s_b, vt_b = torch.linalg.svd(b_gpu, full_matrices=False)
                u_a, s_a, vt_a = torch.linalg.svd(a_gpu, full_matrices=False)
                p = torch.diag(s_b) @ (vt_b @ u_a) @ torch.diag(s_a)
            else:
                # Spectral variant: QR on stacked A/B, then SVD only on compact core.
                # B = Q_B R_B, A^T = Q_A R_A, so BA = Q_B (R_B R_A^T) Q_A^T.
                q_b, r_b = torch.linalg.qr(b_gpu, mode="reduced")
                q_a, r_a = torch.linalg.qr(a_gpu.T, mode="reduced")
                p = r_b @ r_a.T

            u_p, s_p, vt_p = torch.linalg.svd(p, full_matrices=False)
            s_p = s_p.cpu()
            if debug_svals:
                s_cpu = s_p.detach().cpu()
                nz0 = int((s_cpu > 0).sum().item())
                nz_eps = int((s_cpu > sv_eps).sum().item())
                # Diagnostics to locate structural rank bottlenecks.
                try:
                    rank_a = int(torch.linalg.matrix_rank(a).item())
                except Exception:
                    rank_a = -1
                try:
                    rank_b = int(torch.linalg.matrix_rank(b).item())
                except Exception:
                    rank_b = -1
                try:
                    rank_p = int(torch.linalg.matrix_rank(p).item())
                except Exception:
                    rank_p = -1
                a_row_nz = int((torch.linalg.norm(a, dim=1) > sv_eps).sum().item())
                b_col_nz = int((torch.linalg.norm(b, dim=0) > sv_eps).sum().item())
                if sv_topk is not None and int(sv_topk) > 0:
                    vals = s_cpu[: int(sv_topk)].tolist()
                    sval_str = ", ".join(f"{v:.6e}" for v in vals)
                    print(
                        f"{log_prefix}[SVAL] key={key} | len={s_cpu.numel()} | >0={nz0} | >eps({sv_eps})={nz_eps} | "
                        f"rankA={rank_a} rankB={rank_b} rankP={rank_p} | nz_rows_A={a_row_nz} nz_cols_B={b_col_nz} | "
                        f"max={float(s_cpu.max()):.6e} | min={float(s_cpu.min()):.6e} | first_{int(sv_topk)}=[{sval_str}]"
                    )
                else:
                    vals = s_cpu.tolist()
                    sval_str = ", ".join(f"{v:.6e}" for v in vals)
                    print(
                        f"{log_prefix}[SVAL] key={key} | len={s_cpu.numel()} | >0={nz0} | >eps({sv_eps})={nz_eps} | "
                        f"rankA={rank_a} rankB={rank_b} rankP={rank_p} | nz_rows_A={a_row_nz} nz_cols_B={b_col_nz} | "
                        f"max={float(s_cpu.max()):.6e} | min={float(s_cpu.min()):.6e} | all=[{sval_str}]"
                    )

            if stacked_factorization == "svd":
                u_g = (u_b @ u_p).cpu()
                vt_g = (vt_p @ vt_a).cpu()
            else:
                u_g = (q_b @ u_p).cpu()
                vt_g = (vt_p @ q_a.T).cpu()

            k_opt, rank_meta = self._select_rank_florist(
                s_p,
                p.shape,
                threshold=threshold,
                rank_method=rank_method,
                screenot_k=screenot_k,
                screenot_strategy=screenot_strategy,
            )
            rank_meta["stacked_factorization"] = stacked_factorization
            if rank_method == "threshold":
                print(
                    f"{log_prefix} key={key}, method=threshold, factorization={stacked_factorization}, "
                    f"stacked_rank={a.shape[0]}, k_opt={k_opt}, threshold={threshold}"
                )
            elif rank_method == "screenot":
                print(
                    f"{log_prefix} key={key}, method=screenot, factorization={stacked_factorization}, stacked_rank={a.shape[0]}, "
                    f"k_opt={k_opt}, tau={rank_meta.get('tau', 0.0):.6f}, "
                    f"strategy={rank_meta.get('strategy')}, k_used={rank_meta.get('k_used')}"
                )
            else:
                print(
                    f"{log_prefix} key={key}, method=gavish_donoho, factorization={stacked_factorization}, stacked_rank={a.shape[0]}, "
                    f"k_opt={k_opt}, tau={rank_meta.get('tau', 0.0):.6f}, beta={rank_meta.get('beta', 0.0):.4f}"
                )
            layer_idx, matrix_name = self._extract_layer_and_matrix_name(key)
            rank_details.append({
                "key": key,
                "layer": layer_idx,
                "matrix": matrix_name,
                "stacked_rank": int(a.shape[0]),
                "optimal_rank": int(k_opt),
                "rank_method": rank_meta.get("method"),
                "aggregation": aggregation_name,
                "stacked_factorization": stacked_factorization,
                "rank_meta": rank_meta,
            })

            # Eval state: keep strict FLoRIST-optimal rank.
            a_eval_2d = vt_g[:k_opt, :]
            b_eval_2d = u_g[:, :k_opt] @ torch.diag(s_p[:k_opt])
            global_params_eval[key] = self._restore_lora_a_shape(a_eval_2d, a_ref)
            global_params_eval[b_key] = self._restore_lora_b_shape(b_eval_2d, b_ref)

            # Train state sent to clients can differ by init mode.
            if train_init_mode == "trained_a":
                # Keep A at transmit max-rank (trained rows), while B keeps only
                # optimal-rank content and zero-pads remaining columns.
                a_train_2d = vt_g[:train_transmit_rank, :]
                if a_train_2d.shape[0] < train_transmit_rank:
                    pad_a = torch.zeros(
                        (train_transmit_rank - a_train_2d.shape[0], a_train_2d.shape[1]),
                        dtype=a_train_2d.dtype,
                        device=a_train_2d.device,
                    )
                    a_train_2d = torch.cat([a_train_2d, pad_a], dim=0)

                if b_eval_2d.shape[1] < train_transmit_rank:
                    pad_b = torch.zeros(
                        (b_eval_2d.shape[0], train_transmit_rank - b_eval_2d.shape[1]),
                        dtype=b_eval_2d.dtype,
                        device=b_eval_2d.device,
                    )
                    b_train_2d = torch.cat([b_eval_2d, pad_b], dim=1)
                else:
                    b_train_2d = b_eval_2d[:, :train_transmit_rank]

                global_params_train[key] = self._restore_lora_a_shape(a_train_2d, a_ref)
                global_params_train[b_key] = self._restore_lora_b_shape(b_train_2d, b_ref)
            else:
                global_params_train[key] = global_params_eval[key]
                global_params_train[b_key] = global_params_eval[b_key]

        # Homogenize only after FLoRIST truncation so model load is always valid.
        # florist_eval_state = copy.deepcopy(global_params_eval)
        # self.florist_eval_state = copy.deepcopy(florist_eval_state)
        self.florist_eval_state = copy.deepcopy(global_params_eval)  # one copy, stored on self  ~Tanishk
        florist_eval_state = self.florist_eval_state  
        self.florist_eval_rank = self._max_lora_rank_in_state(florist_eval_state)
        self.florist_train_state = copy.deepcopy(global_params_train)
        self.latest_florist_rank_record = {
            "round": int(round_idx + 1) if round_idx is not None else None,
            "aggregation": aggregation_name,
            "stacked_factorization": stacked_factorization,
            "rank_method": rank_method,
            "threshold": float(threshold),
            "screenot_k": int(screenot_k) if screenot_k is not None else -1,
            "screenot_strategy": screenot_strategy,
            "train_init_mode": train_init_mode,
            "train_transmit_rank": int(train_transmit_rank),
            "rank_details": rank_details,
        }
        # Keep raw optimal-rank global adapters; for server bookkeeping only,
        # materialize at max observed rank (no truncation to self.rank).
        materialize_rank = max(1, int(self.florist_eval_rank))
        self._ensure_global_rank(materialize_rank)
        materialized_eval = self._homogenize_state_to_rank(global_params_eval, materialize_rank)
        self.global_model.load_state_dict(materialized_eval, strict=False)
        return self._compute_comm_stats(florist_eval_state)

    def aggregate_spectral(
        self,
        client_parameters,
        threshold=0.9,
        rank_method="threshold",
        screenot_k=-1,
        screenot_strategy="i",
        client_ranks=None,
        global_max_rank=None,
        train_init_mode="zero",
        round_idx=None,
        client_weights=None,
        debug_svals=False,
        sv_topk=20,
        sv_eps=1e-8,
    ):
        return self.aggregate_florist(
            client_parameters=client_parameters,
            threshold=threshold,
            rank_method=rank_method,
            screenot_k=screenot_k,
            screenot_strategy=screenot_strategy,
            client_ranks=client_ranks,
            global_max_rank=global_max_rank,
            train_init_mode=train_init_mode,
            round_idx=round_idx,
            client_weights=client_weights,
            debug_svals=debug_svals,
            sv_topk=sv_topk,
            sv_eps=sv_eps,
            aggregation_name="spectral",
            stacked_factorization="qr",
        )

    def _aggregate_lora_parameters(self, client_parameters, client_weights=None):
        weights = self._normalize_weights(client_parameters, client_weights)
        lora_A_sum = {}
        lora_B_sum = {}
        product_sum = {}

        for client_params, w in zip(client_parameters, weights):
            for key in client_params.keys():
                if 'lora_A' not in key:
                    continue
                b_key = key.replace('lora_A', 'lora_B')
                if b_key not in client_params:
                    continue
                lora_A = client_params[key]
                lora_B = client_params[b_key]

                lora_A_flat = lora_A.view(lora_A.size(0), -1)
                lora_B_flat = lora_B.view(lora_B.size(0), -1)
                if lora_B_flat.size(1) != lora_A_flat.size(0):
                    continue
                product = torch.matmul(lora_B_flat, lora_A_flat)

                if key not in lora_A_sum:
                    lora_A_sum[key] = lora_A * w
                    lora_B_sum[b_key] = lora_B * w
                    product_sum[key] = product * w
                else:
                    lora_A_sum[key] += lora_A * w
                    lora_B_sum[b_key] += lora_B * w
                    product_sum[key] += product * w

        avg_lora_A = {key: lora_A_sum[key] for key in lora_A_sum.keys()}
        avg_lora_B = {key: lora_B_sum[key] for key in lora_B_sum.keys()}
        avg_products = {key: product_sum[key] for key in product_sum.keys()}
        return avg_lora_A, avg_lora_B, avg_products

    def _optimize_delta_lora_B(self, hat_lora_A, hat_lora_B, avg_client_product, num_iterations=10, learning_rate=0.01, lambda_reg=1.0):
        device = next(self.global_model.parameters()).device

        # Move inputs to GPU for the optimization loop
        hat_lora_A_dev       = {k: v.to(device) for k, v in hat_lora_A.items()}
        hat_lora_B_dev       = {k: v.to(device) for k, v in hat_lora_B.items()}
        avg_client_product_dev = {k: v.to(device) for k, v in avg_client_product.items()}

        delta_lora_B_dict = {}
        optimizer_params = []

        for key in hat_lora_B_dev.keys():
            delta_lora_B = torch.empty_like(hat_lora_B_dev[key])
            torch.nn.init.xavier_uniform_(delta_lora_B)
            delta_lora_B.requires_grad = True
            delta_lora_B_dict[key] = delta_lora_B
            optimizer_params.append(delta_lora_B)

        optimizer = optim.SGD(optimizer_params, lr=learning_rate)
        prev_loss = float('inf')
        patience, patience_counter = 5, 0  # early stopping
        for iteration in range(num_iterations):
            optimizer.zero_grad()
            total_loss = 0
            for key in delta_lora_B_dict.keys():
                a_key = key.replace('lora_B', 'lora_A')
                if a_key not in hat_lora_A_dev or a_key not in avg_client_product_dev:
                    continue

                corrected_lora_B = hat_lora_B_dev[key] + delta_lora_B_dict[key]
                lora_A_flattened = hat_lora_A_dev[a_key].view(hat_lora_A_dev[a_key].size(0), -1)
                corrected_lora_B_flattened = corrected_lora_B.view(corrected_lora_B.size(0), -1)
                if corrected_lora_B_flattened.size(1) != lora_A_flattened.size(0):
                    continue

                reconstructed_product = torch.matmul(corrected_lora_B_flattened, lora_A_flattened)
                cosine_similarity = F.cosine_similarity(avg_client_product_dev[a_key], reconstructed_product, dim=0)
                # changed dim from o to 1, to test an issue ~ Tanishk
                loss_term1 = 1 - cosine_similarity.mean()
                loss_term2 = lambda_reg * torch.norm(delta_lora_B_dict[key]) ** 2
                total_loss += loss_term1 + loss_term2

            total_loss.backward()
            optimizer.step()
            loss_val = float(total_loss.detach())
            if iteration % 50 == 0:
                print(f'[LoRA-FAIR refine] iter={iteration}, loss={loss_val:.6f}')

            # Early stopping: stop if loss barely changing
            if abs(prev_loss - loss_val) < 1e-6:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'[LoRA-FAIR refine] Converged at iter={iteration}, loss={loss_val:.6f}')
                    break
            else:
                patience_counter = 0
                prev_loss = loss_val

        # Move results back to CPU (global_params addition happens on CPU)
        self.delta_lora_B = {k: v.detach().cpu() for k, v in delta_lora_B_dict.items()}

    def aggregate_lora_fair(self, client_parameters, num_iterations=1000, learning_rate=0.01, lambda_reg=1.0, client_ranks=None, client_weights=None):
        # LoRA-FAIR needs shape-compatible LoRA tensors across clients.
        # Under heterogeneous ranks, map all client adapters to max selected rank.
        target_rank = max(client_ranks) if client_ranks else self.rank
        self._ensure_global_rank(target_rank)
        client_parameters_h = [self._homogenize_state_to_rank(cp, target_rank) for cp in client_parameters]
        print(f"[LoRA-FAIR] Aggregating {len(client_parameters_h)} clients at common rank={target_rank}")

        hat_lora_A, hat_lora_B, avg_client_product = self._aggregate_lora_parameters(
            client_parameters_h,
            client_weights=client_weights,
        )
        self._optimize_delta_lora_B(hat_lora_A, hat_lora_B, avg_client_product, num_iterations, learning_rate, lambda_reg)

        global_params = self._average_states(client_parameters_h, client_weights=client_weights)
        # changed self.x to self._average_states due to error during training ~Tanishk
        for key in global_params.keys():
            if key in self.delta_lora_B:
                global_params[key] = global_params[key] + self.delta_lora_B[key]
        self.global_model.load_state_dict(global_params, strict=False)
        return self._compute_comm_stats(global_params)

    def get_trainable_parameters(self):
        return {
            k: v for k, v in self.global_model.state_dict().items()
            if any(keyword in k for keyword in ['lora', 'head', 'Prompt'])
        }

    def get_base_parameters(self):
        return copy.deepcopy(self.global_base_state)

    def compute_ablation_svals(
        self,
        client_parameters,
        client_weights=None,
        screenot_k=-1,
        screenot_strategy="i",
    ):
        """Compute singular values (spectral/QR method) and all four threshold cuts per layer.

        Returns a dict keyed by lora_A parameter name, containing:
          - "layer", "matrix": parsed from key
          - "singular_values": list of singular values (descending)
          - "stacked_rank": number of rows in stacked A (= sum of client ranks)
          - "p_shape": shape of the core matrix p
          - "cuts": {
                "energy_09": k,   # energy threshold @ 0.9
                "energy_99": k,   # energy threshold @ 0.99
                "gavish_donoho": k,
                "screenot": k,
            }
          - "gavish_donoho_meta": metadata dict from Gavish-Donoho
          - "screenot_meta": metadata dict from ScreeNOT
        """
        weights = self._normalize_weights(client_parameters, client_weights)
        ablation = {}

        keys = [k for k in client_parameters[0].keys() if "lora_A" in k]
        for key in keys:
            b_key = key.replace("lora_A", "lora_B")
            a_chunks, b_chunks = [], []
            expected_dw = None
            for cp, w in zip(client_parameters, weights):
                if b_key not in cp:
                    continue
                a2 = self._reshape_lora_a_for_svd(cp[key])
                b2 = self._reshape_lora_b_for_svd(cp[b_key])
                if a2 is None or b2 is None or b2.shape[1] != a2.shape[0]:
                    continue
                a_chunks.append(a2 * w)
                b_chunks.append(b2)
                # Accumulate expected delta W = Σ_i w_i * B_i @ A_i
                contrib = b2 @ (a2 * w)
                expected_dw = contrib if expected_dw is None else expected_dw + contrib
            if not a_chunks:
                continue

            a = torch.cat(a_chunks, dim=0)
            b = torch.cat(b_chunks, dim=1)

            # Spectral method: QR factorization of stacked A and B, then SVD of compact core.
            q_b, r_b = torch.linalg.qr(b, mode="reduced")
            q_a, r_a = torch.linalg.qr(a.T, mode="reduced")
            p = r_b @ r_a.T
            _, s_p, _ = torch.linalg.svd(p, full_matrices=False)

            # SVD of expected aggregated delta W
            _, s_edw, _ = torch.linalg.svd(
                expected_dw.to(dtype=torch.float32), full_matrices=False
            )

            # Four threshold cuts
            k_09, _ = self._select_rank_florist(s_p, p.shape, threshold=0.9, rank_method="threshold")
            k_99, _ = self._select_rank_florist(s_p, p.shape, threshold=0.99, rank_method="threshold")
            k_gd, meta_gd = self._select_rank_florist(s_p, p.shape, rank_method="gavish_donoho")
            k_sn, meta_sn = self._select_rank_florist(
                s_p, p.shape, rank_method="screenot",
                screenot_k=screenot_k, screenot_strategy=screenot_strategy,
            )

            layer_idx, matrix_name = self._extract_layer_and_matrix_name(key)
            ablation[key] = {
                "layer": layer_idx,
                "matrix": matrix_name,
                "stacked_rank": int(a.shape[0]),
                "p_shape": [int(p.shape[0]), int(p.shape[1])],
                "singular_values": s_p.detach().cpu().tolist(),
                "expected_deltaw_singular_values": s_edw.detach().cpu().tolist(),
                "cuts": {
                    "energy_09": int(k_09),
                    "energy_99": int(k_99),
                    "gavish_donoho": int(k_gd),
                    "screenot": int(k_sn),
                },
                "gavish_donoho_meta": meta_gd,
                "screenot_meta": meta_sn,
            }

        return ablation

    def get_eval_model_for_flora(self):
        # In stacked-only FLoRA mode, evaluate directly on the stacked global adapter.
        return self.global_model

    def get_eval_model_for_florist(self):
        # Evaluate with raw FLoRIST adapters (post-threshold per-layer ranks),
        # without truncating everything to server.rank.
        if self.florist_eval_state is None:
            return self.global_model
        eval_rank = int(self.florist_eval_rank) if self.florist_eval_rank else self.rank
        eval_rank = max(1, eval_rank)
        model = self._build_lora_model(eval_rank)
        state = self._homogenize_state_to_rank(self.florist_eval_state, eval_rank)
        model.load_state_dict(state, strict=False)
        return model

    def get_eval_model_for_rank(self, target_rank):
        rank = max(1, int(target_rank))
        model = self._build_lora_model(rank)
        state = self._homogenize_state_to_rank(self.global_model.state_dict(), rank)
        model.load_state_dict(state, strict=False)
        return model

    def get_full_parameters(self):
        return self.global_model.state_dict()

    def get_florist_train_parameters(self):
        if self.florist_train_state is not None:
            return copy.deepcopy(self.florist_train_state)
        return self.global_model.state_dict()

    def get_latest_florist_rank_record(self):
        return copy.deepcopy(self.latest_florist_rank_record)
