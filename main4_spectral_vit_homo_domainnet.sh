#!/bin/bash
set -oe pipefail
source lora-fair/bin/activate

# ── Helper functions ──────────────────────────────────────────────────
sanitize() {
  echo "$1" | sed 's/[\/ .]/_/g'
}

# ── Fixed experiment parameters ───────────────────────────────────────
DATASET="domainnet"
MODEL="ViT_L"
AGG="spectral"
MODE="homo"
ROUNDS="50"
CLIENTS="100"
FRAC="0.1"

# ── Init methods and GPUs ─────────────────────────────────────────────
INIT_METHODS=(normal_a)   # add more as needed
GPUS=(0)

OUTBASE="results/main4_spectral_vit_homo_domainnet/screenot/vit_l/domainnet"
LOGBASE="results/main4_spectral_vit_homo_domainnet/screenot/vit_l/domainnet/logs"
mkdir -p "$OUTBASE" "$LOGBASE"

export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1 

for i in "${!INIT_METHODS[@]}"; do
  INIT="${INIT_METHODS[$i]}"
  GPU="${GPUS[$i]}"
  TS="$(date +%Y%m%d_%H%M%S)"
  LOGFILE="$LOGBASE/run_$(sanitize "$DATASET")_$(sanitize "$MODEL")_$(sanitize "$AGG")_${MODE}_r${ROUNDS}_c${CLIENTS}_f$(sanitize "$FRAC")_init_${INIT}_${TS}.log"

  CUDA_VISIBLE_DEVICES=$GPU nohup python3 main.py \
    --dataset         "$DATASET" \
    --data_path       /projects/bewi/hramesh/SpecTraL/datasets/DomainNet \
    --model           "$MODEL" \
    --num_classes     345 \
    --clients         "$CLIENTS" \
    --client_fraction "$FRAC" \
    --lora_rank       32 \
    --florist_rank_method screenot \
    --rounds          "$ROUNDS" \
    --local_epochs    1 \
    --aggregation     "$AGG" \
    --learning_rate   0.01 \
    --batch_size      16 \
    --num_workers     0 \
    --alpha           0.5 \
    --eval_every      5 \
    --florist_pad_init "$INIT" \
    --deltaw_sanity \
    --output_dir      "$OUTBASE/${INIT}" \
    > "$LOGFILE" 2>&1 &

  echo "Launched init=${INIT} on GPU ${GPU} | PID: $! | Log: ${LOGFILE}"

  # Stagger launches by 300s — lets each experiment finish ViT-L weight
  # loading and 100-client DataLoader construction before the next starts.
  if [ $i -lt $((${#INIT_METHODS[@]} - 1)) ]; then
    echo "Waiting 60s before next launch..."
    sleep 300
  fi
done

echo "All ${#INIT_METHODS[@]} experiments launched."