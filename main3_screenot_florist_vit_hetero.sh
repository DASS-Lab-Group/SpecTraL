#!/bin/bash
set -oe pipefail

source lora-fair/bin/activate

# ── Helper functions ──────────────────────────────────────────────────
sanitize() {
  echo "$1" | sed 's/[\/ .]/_/g'
}

# ── Fixed experiment parameters ───────────────────────────────────────
DATASET="nicopp"
MODEL="ViT"
AGG="florist"
MODE="homo"  
ROUNDS="100"
CLIENTS="100"
FRAC="0.1"

# ── 5 init methods on 5 GPUs ──────────────────────────────────────────
INIT_METHODS=(normal_a orthogonal_a)
GPUS=(0 1)

for i in "${!INIT_METHODS[@]}"; do
  INIT="${INIT_METHODS[$i]}"
  GPU="${GPUS[$i]}"
  TS="$(date +%Y%m%d_%H%M%S)"
  LOGFILE="results/main3_spectra_hetero_vit_nicopp/screenot_heavy_tail_strong/logs/run_$(sanitize "$DATASET")_$(sanitize "$MODEL")_$(sanitize "$AGG")_${MODE}_r${ROUNDS}_c${CLIENTS}_f$(sanitize "$FRAC")_init_${INIT}_${TS}.log"

  CUDA_VISIBLE_DEVICES=$GPU nohup python3 main.py \
    --dataset "$DATASET" \
    --data_path /projects/bewi/hramesh/SpecTraL/datasets/Nico++ \
    --model "$MODEL" \
    --num_classes 60 \
    --clients "$CLIENTS" \
    --client_fraction "$FRAC" \
    --rounds "$ROUNDS" \
    --local_epochs 1 \
    --aggregation "$AGG" \
    --learning_rate 0.01 \
    --batch_size 32 \
    --alpha 0.5 \
    --eval_every 5 \
    --heter \
    --heter_rank_profile heavy_tail_strong \
    --florist_rank_method screenot \
    --florist_pad_init "$INIT" \
    --deltaw_sanity \
    --output_dir "results/main3_spectra_hetero_vit_nicopp/screenot_heavy_tail_strong/${INIT}" \
    > "$LOGFILE" 2>&1 &

  echo "Launched init=${INIT} on GPU ${GPU} | PID: $! | Log: ${LOGFILE}"
done

echo "All 2 experiments launched in parallel!"