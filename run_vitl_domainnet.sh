#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_vitl_domainnet.sh
# Launch two ViT-L + DomainNet spectral experiments safely in parallel.
#   Exp 1: spectral / screenot / normal_a  padding  → GPU 0
#   Exp 2: spectral / screenot / orthogonal_a padding → GPU 1
#
# Prerequisites:
#   1. Apply patches first:
#        python apply_vitl_domainnet_patch.py --domainnet DomainNet.py --main main.py
#   2. Start memory monitor (separate terminal or tmux pane):
#        nohup python memory_monitor.py --interval 10 --log mem_vitl.log &
#
# Run:
#   bash run_vitl_domainnet.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────
REPO="/projects/bewi/hramesh/SpecTraL"          # adjust if your repo lives elsewhere
DATA="/data/DomainNet"          # adjust to your DomainNet dataset path
LOGDIR="$REPO/logs/vitl_domainnet"
OUTDIR="$REPO/results/vitl_domainnet"

mkdir -p "$LOGDIR" "$OUTDIR"

# ── Common hyperparameters ────────────────────────────────────────────────
# Matching Table 2 paper config for DomainNet feature+label non-IID
COMMON=(
    --dataset         domainnet
    --data_path       "$DATA"
    --model           ViT_L
    --num_classes     100
    --clients         30
    --client_fraction 0.6
    --rounds          75
    --local_epochs    5
    --lora_rank       32
    --batch_size      16        # reduced from 32: ViT-L activations are 2x larger
    --learning_rate   0.01
    --alpha           0.5
    --aggregation     spectral
    --florist_rank_method screenot
    --florist_screenot_strategy i
    --num_workers     2         # safe for parallel ViT-L runs; avoids shm explosion
    --eval_every      5
    --seed            42
)

# ── Experiment 1: ScreeNOT + normal_a padding ─────────────────────────────
EXP1_NAME="vitl_domainnet_spectral_screenot_normala_30c_75r_r32"
echo "[$(date '+%H:%M:%S')] Launching EXP1: spectral/screenot/normal_a on GPU 0 ..."

CUDA_VISIBLE_DEVICES=0 nohup python "$REPO/main.py" \
    "${COMMON[@]}" \
    --florist_pad_init  normal_a \
    --output_dir        "$OUTDIR/${EXP1_NAME}" \
    > "$LOGDIR/${EXP1_NAME}.log" 2>&1 &

PID1=$!
echo "  EXP1 PID: $PID1  log: $LOGDIR/${EXP1_NAME}.log"

# ── Stagger launch by 60 seconds ─────────────────────────────────────────
# Reason: both experiments call timm.create_model for ViT-L at startup and
# download / cache pretrained weights. Simultaneous downloads + 100-client
# DataLoader init spikes CPU RAM and /dev/shm. 60s lets EXP1 finish its
# model load and DataLoader construction before EXP2 begins.
echo "[$(date '+%H:%M:%S')] Waiting 60s before launching EXP2 (staggered init)..."
sleep 60

# ── Experiment 2: ScreeNOT + orthogonal_a padding ─────────────────────────
EXP2_NAME="vitl_domainnet_spectral_screenot_orthogonala_30c_75r_r32"
echo "[$(date '+%H:%M:%S')] Launching EXP2: spectral/screenot/orthogonal_a on GPU 1 ..."

CUDA_VISIBLE_DEVICES=1 nohup python "$REPO/main.py" \
    "${COMMON[@]}" \
    --florist_pad_init  orthogonal_a \
    --output_dir        "$OUTDIR/${EXP2_NAME}" \
    > "$LOGDIR/${EXP2_NAME}.log" 2>&1 &

PID2=$!
echo "  EXP2 PID: $PID2  log: $LOGDIR/${EXP2_NAME}.log"

# ── Monitor ───────────────────────────────────────────────────────────────
echo ""
echo "Both experiments launched."
echo "  EXP1 PID=$PID1  (GPU 0)  spectral/screenot/normal_a"
echo "  EXP2 PID=$PID2  (GPU 1)  spectral/screenot/orthogonal_a"
