#!/bin/bash
set -oe pipefail

source lora-fair/bin/activate

# ── Helper functions ──────────────────────────────────────────────────
sanitize() {
  echo "$1" | sed 's/[\/ .]/_/g'
}

# hetero settings

# last three arguments are LoRA FAIR Specific
 # --refinement_iterations 5000 \
 #    --refinement_lr 0.01 \
 #    --lambda_reg 0.01 \

COMMON_ARGS="
    --dataset nicopp \
    --data_path /projects/bewi/hramesh/SpecTraL/datasets/Nico++ \
    --model ViT \
    --num_classes 60 \
    --clients 100 \
    --client_fraction 0.1 \
    --rounds 75 \
    --local_epochs 1 \
    -- lora_rank 32 \
    --learning_rate 0.01 \
    --batch_size 32 \
    --alpha 0.5 \
    --save_model \
    --eval_every 5 \
"

OUTBASE="results/main1_nicopp_100c_75r/baseline_hetero_heavy_tail_strong/"
LOGBASE="results/main1_nicopp_100c_75r/baseline_hetero_heavy_tail_strong/logs"

mkdir -p "$OUTBASE" "$LOGBASE"

CUDA_VISIBLE_DEVICES=2 nohup python main.py $COMMON_ARGS \
    --aggregation flex \
    --output_dir "$OUTBASE" \
    > "$LOGBASE/flex_nicopp_100c_75r_hetero_heavy_tail_strong.log" 2>&1 &
echo "flex PID: $!"

CUDA_VISIBLE_DEVICES=3 nohup python main.py $COMMON_ARGS \
    --aggregation florist \
    --output_dir "$OUTBASE" \
    > "$LOGBASE/florist_nicopp_100c_75r_hetero_heavy_tail_strong.log" 2>&1 &
echo "florist PID: $!"

# CUDA_VISIBLE_DEVICES=2 nohup python main.py $COMMON_ARGS \
#     --aggregation flora \
#     --output_dir "$OUTBASE" \
#     > "$LOGBASE/flora_nicopp_100c_75r_hetero.log" 2>&1 &
# echo "flora PID: $!"

CUDA_VISIBLE_DEVICES=4 nohup python main.py $COMMON_ARGS \
    --aggregation fedit \
    --output_dir "$OUTBASE" \
    > "$LOGBASE/fedit_nicopp_100c_75r_heavy_tail_strong.log" 2>&1 &
echo "fedit PID: $!"

# CUDA_VISIBLE_DEVICES=4 nohup python main.py $COMMON_ARGS \
#     --aggregation ffa \
#     --output_dir "$OUTBASE" \
#     > "$LOGBASE/ffa_nicopp_100c_75r_hetero.log" 2>&1 &  
# echo "ffa PID: $!"

# CUDA_VISIBLE_DEVICES=5 nohup python main.py $COMMON_ARGS \
#     --aggregation lora_fair \
#     --output_dir "$OUTBASE" \
#     > "$LOGBASE/lora_fair_nicopp_100c_75r_hetero.log" 2>&1 &  
# echo "lora_fair PID: $!"

echo "All 3 methods launched in parallel!"