#!/bin/bash
set -eo pipefail

source lora-fair/bin/activate

# ── Helper functions ──────────────────────────────────────────────────
sanitize() {
  echo "$1" | sed 's/[\/ .]/_/g'
}

# /projects/bewi/hramesh/SpecTraL/datasets/Nico++
# /projects/bewi/hramesh/SpecTraL/datasets/DomainNet

COMMON_ARGS="
    --dataset nicopp \
    --data_path /projects/bewi/hramesh/SpecTraL/datasets/Nico++ \
    --model ViT \
    --num_classes 60 \
    --clients 100 \
    --client_fraction 0.1 \
    --rounds 100 \
    --local_epochs 1 \
    --lora_rank 32 \
    --learning_rate 0.01 \
    --batch_size 32 \
    --alpha 0.5 \
    --eval_every 5 \
    --heter \
    --heter_rank_profile  uniform \
"

OUTBASE="results/main4_spectral_vitb_hetero/nicopp/baselines"
LOGBASE="results/main4_spectral_vitb_hetero/nicopp/baselines/logs"

mkdir -p "$OUTBASE" "$LOGBASE"

CUDA_VISIBLE_DEVICES=5 nohup python3 main.py $COMMON_ARGS \
    --aggregation flex \
    --output_dir "$OUTBASE" \
    > "$LOGBASE/flex_100c_75r_heter_r32_nicopp_vitb.log" 2>&1 &
echo "flex PID: $!"

echo "[$(date '+%H:%M:%S')] Waiting 60s before launching EXP2 (staggered init)..."
sleep 300

CUDA_VISIBLE_DEVICES=6 nohup python3 main.py $COMMON_ARGS \
    --aggregation florist \
    --output_dir "$OUTBASE" \
    > "$LOGBASE/florist_100c_75r_heter_r32_nicopp_vitb.log" 2>&1 &
echo "florist PID: $!"

# echo "[$(date '+%H:%M:%S')] Waiting 60s before launching EXP2 (staggered init)..."
# sleep 500

# CUDA_VISIBLE_DEVICES=2 nohup python3 main.py $COMMON_ARGS \
#     --aggregation flora \
#     --output_dir "$OUTBASE" \
#     > "$LOGBASE/flora_100c_75r_heter_r32_nicopp_vitl.log" 2>&1 &
# echo "flora PID: $!"

# echo "[$(date '+%H:%M:%S')] Waiting 60s before launching EXP2 (staggered init)..."
# sleep 500

# CUDA_VISIBLE_DEVICES=3 nohup python3 main.py $COMMON_ARGS \
#     --aggregation hetlora \
#     --output_dir "$OUTBASE" \
#     > "$LOGBASE/hetlora_100c_75r_heter_r32_nicopp_vitl.log" 2>&1 &
# echo "hetlora PID: $!"

# # echo "[$(date '+%H:%M:%S')] Waiting 60s before launching EXP2 (staggered init)..."
# # sleep 500

# # CUDA_VISIBLE_DEVICES=5 nohup python main.py $COMMON_ARGS \
# #     --aggregation ffa \
# #     --output_dir "$OUTBASE" \
# #     > "$LOGBASE/ffa_100c_75r_homo_r32_domainnet_vitl.log" 2>&1 &  
# # echo "ffa PID: $!"

# # echo "[$(date '+%H:%M:%S')] Waiting 60s before launching EXP2 (staggered init)..."
# # sleep 500

# # CUDA_VISIBLE_DEVICES=6 nohup python main.py $COMMON_ARGS \
# #     --aggregation lora_fair \
# #     --refinement_iterations 500 \
# #     --refinement_lr 0.01 \
# #     --lambda_reg 0.01 \
# #     --output_dir "$OUTBASE" \
# #     > "$LOGBASE/lora_fair_100c_75r_homo_r32_domainnet_vitl.log" 2>&1 &  
# # echo "lora_fair PID: $!"

echo "All 2 methods launched in parallel!"