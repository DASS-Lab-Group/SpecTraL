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
ROUNDS="75"
CLIENTS="100"
FRAC="0.1"

# ── 5 init methods on 5 GPUs ──────────────────────────────────────────
INIT_METHODS=(orthogonal_a zero normal_a trained_a svd_w0_a_vsqrt)
GPUS=(0 1 2 3 4)

for i in "${!INIT_METHODS[@]}"; do
  INIT="${INIT_METHODS[$i]}"
  GPU="${GPUS[$i]}"
  TS="$(date +%Y%m%d_%H%M%S)"
  LOGFILE="results/main3_spectra_hetero_vit_nicopp/gavishdonoho/logs/run_$(sanitize "$DATASET")_$(sanitize "$MODEL")_$(sanitize "$AGG")_${MODE}_r${ROUNDS}_c${CLIENTS}_f$(sanitize "$FRAC")_init_${INIT}_${TS}.log"

  CUDA_VISIBLE_DEVICES=$GPU nohup python3 main.py \
    --dataset "$DATASET" \
    --data_path /projects/bewi/hramesh/SpecTraL/datasets/Nico++ \
    --model "$MODEL" \
    --num_classes 60 \
    --clients "$CLIENTS" \
    --client_fraction "$FRAC" \
    --rounds "$ROUNDS" \
    --local_epochs 3 \
    --max_iterations 10 \
    --aggregation "$AGG" \
    --learning_rate 0.01 \
    --batch_size 32 \
    --alpha 0.5 \
    --eval_every 5 \
    --heter \
    --florist_rank_method gavish_donoho \
    --florist_pad_init "$INIT" \
    --deltaw_sanity \
    --output_dir "results/main3_spectra_hetero_vit_nicopp/${INIT}" \
    > "$LOGFILE" 2>&1 &

  echo "Launched init=${INIT} on GPU ${GPU} | PID: $! | Log: ${LOGFILE}"
done

echo "All 5 experiments launched in parallel!"