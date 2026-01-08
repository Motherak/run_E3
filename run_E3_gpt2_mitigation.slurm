#!/bin/bash
#SBATCH --job-name=E3_level1_mitigation
#SBATCH --time=23:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=E3_level1_mitigation.o%j
#SBATCH --error=E3_level1_mitigation.e%j

set -euo pipefail

echo "[INFO] Host: $(hostname)"
echo "[INFO] JobID: ${SLURM_JOB_ID}"
echo "[INFO] Start: $(date)"

WORKDIR="/home/hpc/b116ba/b117ba41/Masterarbeit/KonkreteVersuche/E3_drayson/model_collapse"
IMG="$WORKDIR/model_collapse_tf440.sif"
RUN_DIR="$WORKDIR/outputs_fixed/gpt2_topk_importance/job_${SLURM_JOB_ID}"

# Node-local temp
TMPBASE="${SLURM_TMPDIR:-/tmp}/${USER}/model_collapse_${SLURM_JOB_ID}"
mkdir -p "$TMPBASE"
mkdir -p "$TMPBASE/home"

# HF caches into tmp
HF_HOME="$TMPBASE/.hf_cache"
TRANSFORMERS_CACHE="$HF_HOME/transformers"
HF_DATASETS_CACHE="$HF_HOME/datasets"
HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE"

cd "$WORKDIR"

# Ensure dataset config exists for Hydra
mkdir -p config/dataset
cat > config/dataset/wikitext2_local.yaml <<'EOF'
path: data/wikitext2
EOF

# Sanity check dataset files
for f in data/wikitext2/train.json data/wikitext2/validation.json data/wikitext2/test.json; do
  if [ ! -f "$f" ]; then
    echo "[ERROR] Missing dataset file: $f"
    exit 1
  fi
done
echo "[INFO] Dataset JSONs found."

# Models live INSIDE container
GPT2_DIR="/opt/models/gpt2"
DET_DIR="/opt/models/modernbert_ai_detection"

mkdir -p "$RUN_DIR"

echo "[INFO] RUN_DIR: $RUN_DIR"
echo "[INFO] Using container: $IMG"
echo "[INFO] TMPBASE: $TMPBASE"

# ---- HARD ISOLATION: prevent host ~/.local leakage ----
APPT_CMD=(apptainer exec --nv --cleanenv --containall --no-home
  --bind "$WORKDIR:$WORKDIR"
  --bind "$TMPBASE:$TMPBASE"
  --env APPTAINERENV_HOME="$TMPBASE/home"
  --env APPTAINERENV_PYTHONNOUSERSITE=1
  --env APPTAINERENV_TOKENIZERS_PARALLELISM=false
  --env APPTAINERENV_WANDB_DISABLED=true
  --env APPTAINERENV_TRANSFORMERS_OFFLINE=1
  --env APPTAINERENV_HF_HUB_OFFLINE=1
  --env APPTAINERENV_HF_HOME="$HF_HOME"
  --env APPTAINERENV_TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
  --env APPTAINERENV_HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
  --env APPTAINERENV_HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"
  "$IMG"
)

# Debug: ensure torch is from container, NOT /home/.../.local
echo "[INFO] Debug torch location..."
"${APPT_CMD[@]}" python - <<'PY'
import torch, sys
print("torch:", torch.__version__)
print("torch file:", torch.__file__)
print("sys.path head:", sys.path[:10])
PY

echo "[INFO] Starting main.py ..."
"${APPT_CMD[@]}" python main.py \
  hydra.run.dir="$RUN_DIR" \
  dataset=wikitext2_local \
  model=gpt2 \
  decoding=top_k \
  detector=modernbert_mage \
  data_selection=importance_sampling \
  num_iterations=10 \
  wandb_disabled=true \
  ++model.model_name_or_path="$GPT2_DIR" \
  ++model.tokenizer_name="$GPT2_DIR" \
  ++detector.model_path="$DET_DIR" \
  ++detector.tokenizer_name="$DET_DIR"

echo "[INFO] Done at $(date)"
echo "[INFO] Outputs: $RUN_DIR"
