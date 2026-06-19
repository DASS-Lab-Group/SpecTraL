#!/bin/bash
set -eo pipefail

GPU_ID="${1:-0}"
shift || true

export CUDA_VISIBLE_DEVICES="$GPU_ID"

INIT="normal_a"

source lora-fair/bin/activate

# python3 main.py --dataset domainnet --model ViT --aggregation fedit --heter --rounds 20 --local_epochs 1 --clients 100 --client_fraction 0.1

# python3 main.py --dataset domainnet --model ViT --aggregation ffa --heter --rounds 1 --local_epochs 1 --clients 6 --client_fraction 0.5

# python3 main.py --dataset domainnet --model ViT --aggregation florist --rounds 20 --local_epochs 1 --clients 100 --client_fraction 0.1

# python3 main.py --dataset domainnet --model ViT --aggregation flora --heter --rounds 1 --local_epochs 1 --clients 6 --client_fraction 0.5

# python3 main.py --dataset domainnet --model ViT --aggregation flex --heter --rounds 20 --local_epochs 1 --clients 100 --client_fraction 0.1

# python3 main.py --dataset domainnet --model ViT --aggregation lora_fair --heter

DEFAULT_ARGS=(
  --dataset domainnet
  # --data_path /projects/bewi/hramesh/SpecTraL/datasets/Nico++
  --model ViT_L
  --aggregation flora
  --rounds 75
  --local_epochs 1
  --clients 100
  --client_fraction 0.1
  --num_workers 4
  --florist_rank_method screenot     # Choices: "threshold", "gavish_donoho", "screenot"
  --florist_pad_init orthogonal_a          # Choices: "zero", "normal_a", "trained_a", "svd_w0_a", "svd_w0_a_vsqrt", "orthogonal_a"
  --eval_every 5
  --threshold 0.95
  --heter
  --heter_rank_profile uniform
  # --num_classes 60
  # --ablation
  # --deltaw_sanity

  # --heter_rank_profile heavy_tail_strong
  # --florist_debug_svals
  # --florist_sv_topk 55
  # --florist_sv_eps 1e-8
)

sanitize() {
  echo "$1" | sed 's/[\/ .]/_/g'
}

get_arg_val() {
  local key="$1"
  local next=0
  for x in "${DEFAULT_ARGS[@]}"; do
    if [ "$next" -eq 1 ]; then
      echo "$x"
      return 0
    fi
    if [ "$x" = "$key" ]; then
      next=1
    fi
  done
  return 1
}

contains_flag() {
  local flag="$1"
  for x in "${DEFAULT_ARGS[@]}"; do
    if [ "$x" = "$flag" ]; then
      return 0
    fi
  done
  return 1
}

DATASET="$(get_arg_val --dataset || echo unknown)"
MODEL="$(get_arg_val --model || echo unknown)"
AGG="$(get_arg_val --aggregation || echo unknown)"
ROUNDS="$(get_arg_val --rounds || echo na)"
CLIENTS="$(get_arg_val --clients || echo na)"
FRAC="$(get_arg_val --client_fraction || echo na)"
MODE="homo"
if contains_flag --heter; then
  MODE="heter"
fi

TS="$(date +%Y%m%d_%H%M%S)"
LOGFILE="run_$(sanitize "$DATASET")_$(sanitize "$MODEL")_$(sanitize "$AGG")_${MODE}_r${ROUNDS}_c${CLIENTS}_f$(sanitize "$FRAC")_${TS}.log"
exec > "$LOGFILE" 2>&1

echo "[$(date)] Starting job on physical GPU ${GPU_ID} (visible as cuda:0 inside process)"
echo "[$(date)] Log file: ${LOGFILE}"

if [ "$#" -gt 0 ]; then
  python3 main.py "$@"
else
  python3 main.py "${DEFAULT_ARGS[@]}"
fi
