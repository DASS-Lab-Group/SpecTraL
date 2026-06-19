"""
Client implementation for federated vision LoRA training.
"""

import torch
import torch.nn as nn
from tqdm import tqdm
import copy                    
from peft import LoraConfig
from utils import FoundationModel

_BASE_LORA_MODEL_CACHE = {}
class Client:
    def __init__(self, dataloader, num_layers=12, num_classes=10, depth_cls=0, modeltype='ViT', rank=8):
        self.dataloader = dataloader
        self.modeltype = modeltype
        self.rank = rank
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.depth_cls = depth_cls

        cache_key = (modeltype, num_layers, num_classes, rank)
        if cache_key not in _BASE_LORA_MODEL_CACHE:
            print(f"[Client] Building and caching base LoRA model for key={cache_key} ...")
            _BASE_LORA_MODEL_CACHE[cache_key] = FoundationModel(
                num_layers,
                num_classes,
                depth_cls,
                modeltype,
                self._build_lora_config(rank),
            )
            print(f"[Client] Cached. Subsequent clients will use deepcopy.")
        
        # Guard: if cache population failed for any reason, fall back to direct build
        if cache_key in _BASE_LORA_MODEL_CACHE:
            self.local_model = copy.deepcopy(_BASE_LORA_MODEL_CACHE[cache_key])
        else:
            self.local_model = FoundationModel(
                num_layers, num_classes, depth_cls,
                modeltype, self._build_lora_config(rank),
            )    # removed .cuda() for training on ViT L ~ Tanishk
        self.last_local_parameters = None

    def _build_lora_config(self, rank):
        if self.modeltype in ('ViT', 'ViT_L'):
            return LoraConfig(
                r=rank,
                lora_alpha=max(8, 2 * rank),
                target_modules=['attn.proj', 'mlp.fc2'],
                lora_dropout=0.1,
                bias="none",
            )
        if self.modeltype == 'mixer':
            return LoraConfig(
                r=rank,
                lora_alpha=max(8, 2 * rank),
                target_modules=['mlp_tokens.fc2', 'mlp_channels.fc2'],
                lora_dropout=0.1,
                bias="none",
            )
        raise ValueError(f"Unsupported model type: {self.modeltype}")

    @staticmethod
    def _infer_base_weight_key(lora_a_key, target_state):
        # Typical PEFT keys:
        #   <prefix>.lora_A.default.weight
        #   <prefix>.base_layer.weight
        # Fallback to <prefix>.weight if needed.
        for suffix in ('.lora_A.default.weight', '.lora_A.weight'):
            if not lora_a_key.endswith(suffix):
                continue
            prefix = lora_a_key[: -len(suffix)]
            for cand in (f'{prefix}.base_layer.weight', f'{prefix}.weight'):
                if cand in target_state and torch.is_tensor(target_state[cand]) and target_state[cand].ndim >= 2:
                    return cand
        return None

    @staticmethod
    def _svd_init_a_pad_rows(lora_a_key, src, ref, target_state, pad_rows, cache, seed_mode="sigma_vt"):
        if pad_rows <= 0:
            return torch.zeros((0,) + tuple(ref.shape[1:]), dtype=src.dtype, device=src.device)

        cache_key = (lora_a_key, tuple(ref.shape), seed_mode)
        if cache_key not in cache:
            base_key = Client._infer_base_weight_key(lora_a_key, target_state)
            cache[cache_key] = None
            if base_key is not None:
                w0 = target_state[base_key].detach().to(dtype=torch.float32)
                w0_2d = w0.reshape(w0.shape[0], -1)
                in_dim = int(ref[0].numel())
                # Try to coerce to [out_dim, in_dim].
                if w0_2d.shape[1] != in_dim and w0_2d.shape[0] == in_dim:
                    w0_2d = w0_2d.t()
                if w0_2d.shape[1] == in_dim:
                    try:
                        # W0 = U * Sigma * V^T, with V^T row-shape matching LoRA A rows.
                        _, s, vh = torch.linalg.svd(w0_2d, full_matrices=False)
                        max_rows = int(ref.shape[0])
                        k = min(max_rows, int(vh.shape[0]), int(s.shape[0]))
                        seed = torch.zeros((max_rows, in_dim), dtype=w0_2d.dtype, device=w0_2d.device)
                        if k > 0:
                            if seed_mode == "sigma_vt":
                                seed[:k, :] = vh[:k, :] * s[:k].unsqueeze(1)
                            elif seed_mode == "sqrt_sigma_vt":
                                s_root = torch.sqrt(torch.clamp(s[:k], min=0.0))
                                seed[:k, :] = vh[:k, :] * s_root.unsqueeze(1)
                            else:
                                raise ValueError(f"Unsupported seed_mode: {seed_mode}")
                        cache[cache_key] = seed
                    except RuntimeError:
                        cache[cache_key] = None

        seed_2d = cache.get(cache_key, None)
        if seed_2d is None:
            return torch.zeros((pad_rows,) + tuple(ref.shape[1:]), dtype=src.dtype, device=src.device)

        pad_2d = seed_2d[:pad_rows, :]
        return pad_2d.to(dtype=src.dtype, device=src.device).reshape((pad_rows,) + tuple(ref.shape[1:]))

    @staticmethod
    def _orthogonal_completion_a_pad_rows(src, ref, pad_rows):
        if pad_rows <= 0:
            return torch.zeros((0,) + tuple(ref.shape[1:]), dtype=src.dtype, device=src.device)

        src_dtype = src.dtype
        src_device = src.device
        src_2d = src.reshape(src.shape[0], -1).to(dtype=torch.float32)
        in_dim = int(ref[0].numel())

        # Build orthonormal basis for existing row-space of A.
        if src_2d.shape[0] > 0:
            exist = src_2d.t()  # [in_dim, r_src]
            try:
                q_exist, _ = torch.linalg.qr(exist, mode='reduced')
            except RuntimeError:
                q_exist = torch.empty((in_dim, 0), dtype=src_2d.dtype, device=src_2d.device)
        else:
            q_exist = torch.empty((in_dim, 0), dtype=src_2d.dtype, device=src_2d.device)

        rank_exist = int(q_exist.shape[1])
        max_perp = max(0, in_dim - rank_exist)
        need_perp = min(int(pad_rows), int(max_perp))

        pad_parts = []
        if need_perp > 0:
            q_perp = None
            for _ in range(3):
                rand_cols = max(need_perp, min(in_dim, need_perp + 8))
                z = torch.randn((in_dim, rand_cols), dtype=src_2d.dtype, device=src_2d.device)
                if rank_exist > 0:
                    z = z - q_exist @ (q_exist.t() @ z)
                try:
                    qz, rz = torch.linalg.qr(z, mode='reduced')
                except RuntimeError:
                    continue
                if rz.ndim == 2:
                    diag = torch.abs(torch.diag(rz))
                    keep = diag > 1e-6
                    qz = qz[:, keep]
                if qz.shape[1] >= need_perp:
                    q_perp = qz[:, :need_perp]
                    break
            if q_perp is None:
                q_perp = torch.zeros((in_dim, need_perp), dtype=src_2d.dtype, device=src_2d.device)
            pad_parts.append(q_perp.t())  # [need_perp, in_dim]

        rem = int(pad_rows) - int(need_perp)
        if rem > 0:
            # Complement exhausted: fallback random normalized rows.
            z = torch.randn((rem, in_dim), dtype=src_2d.dtype, device=src_2d.device)
            z = z / (torch.linalg.norm(z, dim=1, keepdim=True) + 1e-12)
            pad_parts.append(z)

        if pad_parts:
            pad_2d = torch.cat(pad_parts, dim=0)
        else:
            pad_2d = torch.zeros((int(pad_rows), in_dim), dtype=src_2d.dtype, device=src_2d.device)

        # Match typical magnitude of existing A rows to avoid scale shocks.
        if src_2d.shape[0] > 0:
            target_norm = torch.mean(torch.linalg.norm(src_2d, dim=1)).clamp_min(1e-6)
        else:
            target_norm = torch.tensor(1.0 / (in_dim ** 0.5), dtype=src_2d.dtype, device=src_2d.device)
        cur_norm = torch.linalg.norm(pad_2d, dim=1, keepdim=True).clamp_min(1e-12)
        pad_2d = pad_2d * (target_norm / cur_norm)

        return pad_2d.to(dtype=src_dtype, device=src_device).reshape((int(pad_rows),) + tuple(ref.shape[1:]))

    def get_full_parameters(self):
        return self.local_model.state_dict()

    def get_trainable_parameters(self):
        trainable_params = {}
        for name, param in self.local_model.named_parameters():
            if any(keyword in name for keyword in ['lora', 'head', 'Prompt']):
                trainable_params[name] = param
        return trainable_params

    def load_parameters(self, parameters, init_padded_lora_a=False, florist_pad_mode=None):
        if florist_pad_mode is None:
            florist_pad_mode = "normal_a" if init_padded_lora_a else "zero"
        if florist_pad_mode not in ("zero", "normal_a", "trained_a", "svd_w0_a", "svd_w0_a_vsqrt", "orthogonal_a"):
            raise ValueError(f"Unsupported florist_pad_mode: {florist_pad_mode}")

        target = self.local_model.state_dict()
        adapted = {}
        svd_cache = {}
        for k, v in parameters.items():
            if k not in target:
                continue
            ref = target[k]
            if ref.shape == v.shape:
                adapted[k] = v
                continue

            # Handle LoRA A: rank on dim 0.
            if 'lora_A' in k and v.ndim >= 2 and ref.ndim == v.ndim and v.shape[1:] == ref.shape[1:]:
                if v.shape[0] >= ref.shape[0]:
                    adapted[k] = v[: ref.shape[0], ...]
                else:
                    pad_rows = int(ref.shape[0] - v.shape[0])
                    if florist_pad_mode == "normal_a":
                        pad = torch.empty((pad_rows,) + tuple(v.shape[1:]), dtype=v.dtype, device=v.device)
                        torch.nn.init.normal_(pad, mean=0.0, std=0.02)
                    elif florist_pad_mode in ("svd_w0_a", "svd_w0_a_vsqrt"):
                        seed_mode = "sigma_vt" if florist_pad_mode == "svd_w0_a" else "sqrt_sigma_vt"
                        pad = self._svd_init_a_pad_rows(
                            lora_a_key=k,
                            src=v,
                            ref=ref,
                            target_state=target,
                            pad_rows=pad_rows,
                            cache=svd_cache,
                            seed_mode=seed_mode,
                        )
                    elif florist_pad_mode == "orthogonal_a":
                        pad = self._orthogonal_completion_a_pad_rows(
                            src=v,
                            ref=ref,
                            pad_rows=pad_rows,
                        )
                    else:
                        # 'zero' and 'trained_a' both use zero fallback padding on client.
                        pad = torch.zeros((pad_rows,) + tuple(v.shape[1:]), dtype=v.dtype, device=v.device)
                    adapted[k] = torch.cat([v, pad], dim=0)
                continue

            # Handle LoRA B: rank on dim 1 (always zero-pad missing columns).
            if 'lora_B' in k and v.ndim >= 2 and ref.ndim == v.ndim and v.shape[0] == ref.shape[0] and v.shape[2:] == ref.shape[2:]:
                if v.shape[1] >= ref.shape[1]:
                    adapted[k] = v[:, : ref.shape[1], ...]
                else:
                    pad_cols = int(ref.shape[1] - v.shape[1])
                    pad = torch.zeros((v.shape[0], pad_cols) + tuple(v.shape[2:]), dtype=v.dtype, device=v.device)
                    adapted[k] = torch.cat([v, pad], dim=1)
            # non-lora mismatches are skipped
        self.local_model.load_state_dict(adapted, strict=False)

    def load_and_prepare_flora_parameters(self, parameters):
        # 1) Build a temporary model at stacked global rank and load full server params.
        stacked_rank = self._infer_rank_from_state(parameters)
        stacked_model = FoundationModel(
            self.num_layers,
            self.num_classes,
            self.depth_cls,
            self.modeltype,
            self._build_lora_config(stacked_rank),
        ) # removed .cuda() for vit l training ~tanishk
        stacked_model.load_state_dict(parameters, strict=False)

        # 2) Merge full stacked adapters into the backbone.
        merged_backbone = stacked_model.backbone.merge_and_unload()

        # 3) Rebuild local-rank LoRA model on top of merged base weights.
        base_model = FoundationModel(self.num_layers, self.num_classes, self.depth_cls, self.modeltype, lora_config=None) # removed .cuda() ~tanishk
        base_model.backbone.load_state_dict(merged_backbone.state_dict(), strict=False)

        fresh_lora_model = FoundationModel(
            self.num_layers,
            self.num_classes,
            self.depth_cls,
            self.modeltype,
            self._build_lora_config(self.rank),
        ) # removed .cuda() ~tanishk
        target = fresh_lora_model.backbone.base_model.model.state_dict()
        merged_state = base_model.backbone.state_dict()
        adapted = {k: v for k, v in merged_state.items() if (k in target and 'lora_' not in k)}
        fresh_lora_model.backbone.base_model.model.load_state_dict(adapted, strict=False)

        self.local_model = fresh_lora_model

    @staticmethod
    def _infer_rank_from_state(state_dict):
        max_rank = 1
        for k, v in state_dict.items():
            if not torch.is_tensor(v) or v.ndim < 2:
                continue
            if 'lora_A' in k:
                max_rank = max(max_rank, int(v.shape[0]))
            elif 'lora_B' in k:
                max_rank = max(max_rank, int(v.shape[1]))
        return max_rank

    def load_base_parameters(self, base_parameters):
        # Load only backbone base weights; keep client LoRA freshly initialized.
        target = self.local_model.backbone.base_model.model.state_dict()
        adapted = {k: v for k, v in base_parameters.items() if k in target and 'lora_' not in k}
        self.local_model.backbone.base_model.model.load_state_dict(adapted, strict=False)

    def train_baseline(self, learning_rate, epochs, max_iterations, method='lora_fair'):
        self.local_model = self.local_model.cuda() # for ViT L training ~Tanishk
        for name, param in self.local_model.named_parameters():
            if any(keyword in name for keyword in ['head', 'lora', 'Prompt']):
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)

        if method == 'ffa':
            for name, param in self.local_model.named_parameters():
                if 'lora_A' in name:
                    torch.nn.init.normal_(param, mean=0.0, std=0.02)
                    param.requires_grad_(False)

        head_params = list(self.local_model.backbone.head.parameters())
        head_param_ids = list(map(id, head_params))
        base_params = [p for p in self.local_model.parameters() if id(p) not in head_param_ids and p.requires_grad]

        criterion = nn.CrossEntropyLoss().cuda()
        optimizer = torch.optim.SGD([
            {'params': base_params, 'lr': learning_rate},
            {'params': head_params, 'lr': learning_rate}
        ], lr=learning_rate, momentum=0.9, weight_decay=1e-5)

    # below code is changed to add loss tracking and logging ~ Tanishk

        total_loss = 0.0
        num_batches = 0

        self.local_model.train()
        for ep in range(epochs):
            pbar = tqdm(self.dataloader, desc=f"Client train epoch {ep + 1}/{epochs}", leave=False)
            for iteration, (images, labels) in enumerate(pbar):
                # if iteration > max_iterations:           #removed the cap for experimentation ~tanishk
                #     break
                optimizer.zero_grad()
                images, labels = images.cuda(), labels.cuda()
                outputs = self.local_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=10, norm_type=2)
                optimizer.step()

                # Track loss
                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix(loss=f"{loss.item():.4f}")

        self.local_model = self.local_model.cpu()
        self.last_local_parameters = self.local_model.state_dict()  # CPU tensors now
        torch.cuda.empty_cache()
        # Return average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        return avg_loss
