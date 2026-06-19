# SpecTraL: Spectral Transformation for Layer-wise Global Rank Discovery in Federated LoRA for Vision Transformers

**Hariharan Ramesh** and **Jyotikrishna Dass** — Department of Electrical and Computer Engineering, University of Arizona
**ECML PKDD 2026**

This repository contains the official implementation of **SpecTraL**, a federated LoRA aggregation
framework that discovers per-layer global ranks automatically — with no manual threshold tuning —
by analyzing the singular-value spectrum of the aggregated update via Random Matrix Theory.

SpecTraL stacks heterogeneous client LoRA adapters, performs an orthonormal **Householder QR**
transformation directly in the low-rank latent space, and recovers the *exact* singular values of the
aggregated update `ΔW* = Σ_k p_k B_k A_k` from a compact `r × r` core matrix `C = R_B R_Aᵀ` — without
ever forming the dense `m × n` update. It then applies the **ScreeNOT** estimator to separate the
inter-client consensus signal from non-IID noise, yielding an MSE-optimal, layer-adaptive global rank
`r*`. A **padding-aware initialization** scheme then maps the compact global adapters back to each
client's full local rank for the next round of training.

> 📄 The accompanying paper is included in this repository for reference. Please cite it if you use this code (see [Citation](#citation)).

---

## Method at a glance

Per communication round, for each adapted layer:

1. **Spectral transformation (Householder QR).** Stack local adapters
   `B_stack = [B_1 | … | B_K]`, `A_stack = [p_1 A_1; … ; p_K A_K]` with stacked rank `r = Σ_k r_k`.
   Compute thin QR factorizations `B_stack = Q_B R_B`, `A_stackᵀ = Q_A R_A` and form the
   **core matrix** `C = R_B R_Aᵀ ∈ ℝ^{r×r}`. The singular values of `C` are *exactly* those of `ΔW*`.
2. **Rank discovery (ScreeNOT).** Take the SVD `C = U_C Σ_C V_Cᵀ` and apply ScreeNOT to its singular
   values to obtain the MSE-optimal hard threshold, giving the global rank `r* = |{i : σ_i > θ̂}|`.
   No energy threshold to tune.
3. **Compact global adapters.** Build `B_g`, `A_g` of rank `r*` by applying the stored Householder
   reflectors to the leading `r*` SVD factors (implicit `Q` application via `DORMQR`).
4. **Padding-aware re-initialization.** Each client pads the received rank-`r*` adapters back to its
   local rank `r_k`. With the recommended **`normal_a`** scheme, `A` is padded with Gaussian rows and
   `B` is zero-padded, so the initial forward-pass contribution is preserved exactly while the new
   rows receive immediate gradient signal.

---

## Installation

```bash
# Python 3.8+ recommended
python -m venv lora-fair && source lora-fair/bin/activate

# Core dependencies
pip install -r requirements.txt

# ScreeNOT (required for SpecTraL — NOT in requirements.txt)
pip install screenot
```

> ⚠️ **`screenot` must be installed separately.** It is required whenever
> `--florist_rank_method screenot` is used (i.e., for SpecTraL). If it is missing, the run will raise a
> clear error pointing to the install command.

Models (`vit_base_patch16_224`, `vit_large_patch16_224.augreg_in21k`, `mixer_b16_224`) are pulled from
[`timm`](https://github.com/huggingface/pytorch-image-models) and cached on first use.

---

## Datasets

| Dataset      | `--dataset` | Notes                                                                                  |
|--------------|-------------|----------------------------------------------------------------------------------------|
| DomainNet    | `domainnet` | 6 domains (clipart, infograph, painting, quickdraw, real, sketch). Paper uses first 100 classes (`--num_classes 100`). |
| NICO++       | `nicopp`    | Common-context split, 6 contexts (autumn, dim, grass, outdoor, rock, water), 60 classes (`--num_classes 60`). |

1. Download DomainNet from the [official site](http://ai.bu.edu/M3SDA/) and/or NICO++.
2. Point `--data_path` at your local copy:

```bash
python main.py --dataset domainnet --data_path /path/to/DomainNet --num_classes 100 ...
```

Non-IID partitioning follows LoRA-FAIR: clients are grouped by domain/context, and labels within each
group are further skewed by a Dirichlet distribution with concentration `--alpha` (default `0.5`).

---

## Quick start

### Run SpecTraL (our method)

SpecTraL is the `spectral` aggregation with **ScreeNOT** rank discovery and **`normal_a`** padding:

```bash
python main.py \
    --dataset domainnet --data_path /path/to/DomainNet --num_classes 100 \
    --model ViT --aggregation spectral \
    --florist_rank_method screenot \
    --florist_pad_init normal_a \
    --clients 100 --client_fraction 0.1 --rounds 75 --local_epochs 1 \
    --lora_rank 32 --learning_rate 0.01 --alpha 0.5 --seed 42 \
    --eval_every 5
```

> **The two flags that define SpecTraL:** `--florist_rank_method screenot` and `--florist_pad_init normal_a`.

### Heterogeneous client ranks

Add `--heter` to give clients different LoRA ranks. The rank levels are `{64, 32, 16, 8, 4}`,
distributed across clients by `--heter_rank_profile`:

```bash
python main.py \
    --dataset domainnet --data_path /path/to/DomainNet --num_classes 100 \
    --model ViT_L --aggregation spectral \
    --florist_rank_method screenot --florist_pad_init normal_a \
    --heter --heter_rank_profile heavy_tail_light \
    --clients 100 --client_fraction 0.1 --rounds 75 --local_epochs 1 \
    --eval_every 5
```

You can also pin an explicit per-client rank list with `--local_ranks "64,32,16,8,4"` (cycled/truncated
to the client count), which overrides the profile.

### Convenience launch scripts

- [`run_vit.sh`](run_vit.sh) — single-run template (edit `DEFAULT_ARGS`); auto-names the log file from
  the chosen config. Usage: `bash run_vit.sh <GPU_ID> [extra main.py args]`.
- [`run_vitl_domainnet.sh`](run_vitl_domainnet.sh) — launches two staggered ViT-L / DomainNet SpecTraL
  runs (`normal_a` vs `orthogonal_a` padding) on GPUs 0 and 1.

---

## Reproducing baselines

All methods run through the same `main.py`; only `--aggregation` (and a few method-specific flags)
change.

| Method        | `--aggregation` | Key flags                                                            |
|---------------|-----------------|----------------------------------------------------------------------|
| **SpecTraL**  | `spectral`      | `--florist_rank_method screenot --florist_pad_init normal_a`         |
| FLoRIST       | `florist`       | `--florist_rank_method threshold --threshold 0.9`                    |
| FedIT         | `fedit`         | —                                                                    |
| FFA-LoRA      | `ffa`           | —                                                                    |
| FLoRA         | `flora`         | —                                                                    |
| FlexLoRA      | `flex`          | —                                                                    |
| LoRA-FAIR     | `lora_fair`     | `--refinement_iterations 1000 --refinement_lr 0.01 --lambda_reg 1.0` |
| HetLoRA       | `hetlora`       | use with `--heter`                                                   |

Example (FLoRIST with the paper's practical fixed threshold τ = 0.9):

```bash
python main.py --dataset nicopp --data_path /path/to/NICOPP --num_classes 60 \
    --model ViT --aggregation florist --florist_rank_method threshold --threshold 0.9 \
    --clients 100 --client_fraction 0.1 --rounds 75 --local_epochs 1 --eval_every 5
```

---

## Key arguments

Full descriptions are documented inline at the top of [`main.py`](main.py) (`parse_arguments`). The most
important ones:

#### Data & model
- `--dataset {domainnet,nicopp}` — benchmark.
- `--data_path PATH` — **must** point to your dataset.
- `--model {ViT,ViT_L,mixer}` — backbone (ViT-B/16, ViT-L/16, MLP-Mixer).
- `--num_classes INT` — `100` for DomainNet (paper), `60` for NICO++.

#### Federated setup
- `--clients`, `--client_fraction` — total clients and fraction sampled per round.
- `--rounds`, `--local_epochs` — communication rounds and local epochs.
- `--lora_rank` — local LoRA rank (homogeneous mode; default `32`).
- `--learning_rate`, `--batch_size`, `--alpha` (Dirichlet non-IID), `--seed`.

#### Aggregation
- `--aggregation {lora_fair,fedit,ffa,flora,flex,florist,spectral,hetlora}` — method.

#### Spectral / FLoRIST rank discovery (used by `spectral` and `florist`)
- `--florist_rank_method {threshold,gavish_donoho,screenot}` — rank-selection rule.
  Use **`screenot`** for SpecTraL. `threshold` is FLoRIST's energy criterion (with `--threshold`).
  `gavish_donoho` is implemented but **not part of the paper**.
- `--florist_screenot_k INT` — upper bound `k` on signal rank for ScreeNOT (`-1` = auto).
- `--florist_screenot_strategy {i,w,0}` — ScreeNOT pseudo-noise strategy (default `i`).
- `--florist_pad_init {zero,normal_a,trained_a,svd_w0_a,svd_w0_a_vsqrt,orthogonal_a}` —
  client-side padding/initialization of the `r_k − r*` residual rows of `A`
  (`B` is always zero-padded). Use **`normal_a`** for SpecTraL. These correspond to the initialization
  strategies ablated in the paper (Gaussian = `normal_a`, Zero-padding = `zero`, Orthogonal complement
  = `orthogonal_a`, Pretrained-SVD = `svd_w0_a`, Trained-A = `trained_a`).

#### Heterogeneous ranks
- `--heter` — enable heterogeneous client ranks (levels `{64,32,16,8,4}`).
- `--heter_rank_profile {heavy_tail_light,heavy_tail_strong,uniform}` — distribution over levels.
- `--local_ranks "r1,r2,..."` — explicit per-client ranks (overrides the profile).

#### LoRA-FAIR-specific
- `--refinement_iterations`, `--refinement_lr`, `--lambda_reg` — server-side residual refinement.

#### Diagnostics & output
- `--output_dir` — results root (default `./results`).
- `--eval_every`, `--ablation_every` — evaluation / ablation-save cadence.
- `--ablation` — after round 1, dump per-layer singular values of the core matrix and all four
  threshold cut points (energy@0.9, energy@0.99, Gavish-Donoho, ScreeNOT) to JSON.
- `--deltaw_sanity` / `--deltaw_sanity_topk` — sanity-check the expected weighted-average `ΔW` against
  the realized global adapter.
- `--florist_debug_svals`, `--florist_sv_topk`, `--florist_sv_eps` — singular-value logging.

---

## Output structure

Each run writes to a self-describing subdirectory under `--output_dir`:

```
results/{model}/{dataset}/{aggregation}/
    {mode}_r{rank}_rounds{R}_c{clients}_f{frac}_ep{epochs}_lr{lr}_a{alpha}_s{seed}[_{method}_pad_{pad}]/
        results.json              # accuracies per round / per domain
        optimal_ranks.jsonl       # discovered per-layer r* over rounds (spectral/florist)
        ablation_round{N}.json    # per-layer spectra + cut points (with --ablation)
        model.pth                 # final model (with --save_model)
```

`{mode}` is `heter` or `homo`. The `_{method}_pad_{pad}` suffix is appended only for `florist`/`spectral`.

---

## Analysis & plots

[`hari_plots.ipynb`](hari_plots.ipynb) reads the `ablation_round*.json` files and reproduces the
paper's spectral figures (e.g., scree plots with energy/ScreeNOT cut lines, and the core-matrix vs.
expected-`ΔW` singular-value overlap). Point the `MODEL`/`DATASET`/`SETTING`/`METHOD` config cells at
your run directory.

---

## Repository layout

```
SpecTraL/
├── main.py            # entry point: arg parsing, FL loop, run-dir layout, ablation hooks
├── server.py          # all aggregation strategies + ScreeNOT/Gavish-Donoho rank discovery
├── client.py          # local LoRA training
├── utils.py           # evaluation helpers
├── models/
│   ├── GetModel.py     # timm-backed ViT-B / ViT-L / Mixer factory
│   └── structure*.py   # VPT-ViT / Mixer LoRA-adapted architectures
├── datasets/
│   ├── DomainNet.py    # DomainNet loader + non-IID partition
│   └── NICOPP.py       # NICO++ loader + non-IID partition
├── hari_plots.ipynb   # spectral analysis & paper figures
├── run_vit.sh, run_vitl_domainnet.sh, ...   # launch scripts
└── requirements.txt
```

---

## Citation

If you use this code, please cite SpecTraL:

```bibtex
@inproceedings{ramesh2026spectral,
  title     = {Spectral Transformation for Layer-wise Global Rank Discovery in Federated LoRA for Vision Transformers},
  author    = {Ramesh, Hariharan and Dass, Jyotikrishna},
  booktitle = {European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML PKDD)},
  year      = {2026}
}
```

This work builds on **LoRA-FAIR** and uses the **ScreeNOT** estimator; please also cite them:

```bibtex
@inproceedings{bian2025lorafair,
  title     = {LoRA-FAIR: Federated LoRA Fine-Tuning with Aggregation and Initialization Refinement},
  author    = {Bian, Jieming and Wang, Lei and Zhang, Letian and Xu, Jie},
  booktitle = {ICCV},
  year      = {2025}
}

@article{donoho2023screenot,
  title   = {ScreeNOT: Exact MSE-optimal singular value thresholding in correlated noise},
  author  = {Donoho, David and Gavish, Matan and Romanov, Elad},
  journal = {Annals of Statistics},
  year    = {2023}
}
```

---

## Acknowledgments

This work used DeltaAI at NCSA through ACCESS allocation #CIS250561 (NSF grants #2138259, #2138286,
#2138307, #2137603, #2138296), and the NVIDIA Academic Grant Program (A100 80 GiB GPUs via NVIDIA Brev).
The codebase is derived from [LoRA-FAIR](https://github.com/jmbian/LoRA-FAIR).
