#!/bin/bash
set -oe pipefail

source lora-fair/bin/activate

# ── Helper functions ──────────────────────────────────────────────────
sanitize() {
  echo "$1" | sed 's/[\/ .]/_/g'
}

# /projects/bewi/hramesh/SpecTraL/datasets/Nico++
# /projects/bewi/hramesh/SpecTraL/datasets/DomainNet
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 

COMMON_ARGS="
    --dataset domainnet \
    --data_path /projects/bewi/hramesh/SpecTraL/datasets/DomainNet \
    --model ViT_L \
    --num_classes 345 \
    --clients 100 \
    --client_fraction 0.1 \
    --rounds 50 \
    --local_epochs 1 \
    --num_workers 0 \
    --lora_rank 32 \
    --learning_rate 0.01 \
    --batch_size 16 \
    --alpha 0.5 \
    --save_model \
    --eval_every 5
"

OUTBASE="results/main4_spectral_vit_homo_domainnet/baseline/vit_l/domainnet"
LOGBASE="results/main4_spectral_vit_homo_domainnet/baseline/vit_l/domainnet/logs"

mkdir -p "$OUTBASE" "$LOGBASE"

CUDA_VISIBLE_DEVICES=1 nohup python main.py $COMMON_ARGS \
    --aggregation flex \
    --output_dir "$OUTBASE" \
    > "$LOGBASE/flex_100c_75r_homo_r32_domainnet_vitl.log" 2>&1 &
echo "flex PID: $!"

echo "[$(date '+%H:%M:%S')] Waiting 60s before launching EXP2 (staggered init)..."
sleep 500

CUDA_VISIBLE_DEVICES=2 nohup python main.py $COMMON_ARGS \
    --aggregation florist \
    --output_dir "$OUTBASE" \
    > "$LOGBASE/florist_nicopp_100c_75r_r32_domainnet_vitl.log" 2>&1 &
echo "florist PID: $!"

echo "[$(date '+%H:%M:%S')] Waiting 60s before launching EXP2 (staggered init)..."
sleep 500

CUDA_VISIBLE_DEVICES=3 nohup python main.py $COMMON_ARGS \
    --aggregation flora \
    --output_dir "$OUTBASE" \
    > "$LOGBASE/flora_100c_75r_homo_r32_nicopp_domainnet.log" 2>&1 &
echo "flora PID: $!"

echo "[$(date '+%H:%M:%S')] Waiting 60s before launching EXP2 (staggered init)..."
sleep 500

CUDA_VISIBLE_DEVICES=4 nohup python main.py $COMMON_ARGS \
    --aggregation fedit \
    --output_dir "$OUTBASE" \
    > "$LOGBASE/fedit_100c_75r_homo_r32_domainnet_vitl.log" 2>&1 &
echo "fedit PID: $!"

echo "[$(date '+%H:%M:%S')] Waiting 60s before launching EXP2 (staggered init)..."
sleep 500

CUDA_VISIBLE_DEVICES=5 nohup python main.py $COMMON_ARGS \
    --aggregation ffa \
    --output_dir "$OUTBASE" \
    > "$LOGBASE/ffa_100c_75r_homo_r32_domainnet_vitl.log" 2>&1 &  
echo "ffa PID: $!"

echo "[$(date '+%H:%M:%S')] Waiting 60s before launching EXP2 (staggered init)..."
sleep 500

CUDA_VISIBLE_DEVICES=6 nohup python main.py $COMMON_ARGS \
    --aggregation lora_fair \
    --refinement_iterations 500 \
    --refinement_lr 0.01 \
    --lambda_reg 0.01 \
    --output_dir "$OUTBASE" \
    > "$LOGBASE/lora_fair_100c_75r_homo_r32_domainnet_vitl.log" 2>&1 &  
echo "lora_fair PID: $!"

echo "All 6 methods launched in parallel!"