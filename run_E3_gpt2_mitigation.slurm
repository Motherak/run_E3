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

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------
WORKDIR="/home/hpc/b116ba/b117ba41/Masterarbeit/KonkreteVersuche/E3_drayson/model_collapse"
IMG="$WORKDIR/model_collapse_tf440.sif"
RUN_DIR="$WORKDIR/outputs_fixed/gpt2_topk_importance/job_${SLURM_JOB_ID}"

# ------------------------------------------------------------
# Node-local temp (keep caches off /home, isolate HOME)
# ------------------------------------------------------------
TMPBASE="${SLURM_TMPDIR:-/tmp}/${USER}/model_collapse_${SLURM_JOB_ID}"
mkdir -p "$TMPBASE" "$TMPBASE/home"

HF_HOME="$TMPBASE/.hf_cache"
TRANSFORMERS_CACHE="$HF_HOME/transformers"
HF_DATASETS_CACHE="$HF_HOME/datasets"
HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE"

cd "$WORKDIR"

# ------------------------------------------------------------
# Dataset config for Hydra
# ------------------------------------------------------------
mkdir -p config/dataset
cat > config/dataset/wikitext2_local.yaml <<'EOF'
path: data/wikitext2
EOF

for f in data/wikitext2/train.json data/wikitext2/validation.json data/wikitext2/test.json; do
  if [ ! -f "$f" ]; then
    echo "[ERROR] Missing dataset file: $f"
    exit 1
  fi
done
echo "[INFO] Dataset JSONs found."

# ------------------------------------------------------------
# Models live INSIDE the container
# ------------------------------------------------------------
GPT2_DIR="/opt/models/gpt2"
DET_DIR="/opt/models/modernbert_ai_detection"

mkdir -p "$RUN_DIR"

echo "[INFO] RUN_DIR: $RUN_DIR"
echo "[INFO] Using container: $IMG"
echo "[INFO] TMPBASE: $TMPBASE"
echo "[INFO] WORKDIR: $WORKDIR"

# ------------------------------------------------------------
# HARD ISOLATION: prevent host ~/.local leakage into container
# ------------------------------------------------------------
APPT_CMD=(apptainer exec --nv --cleanenv --containall --no-home
  --bind "$WORKDIR:$WORKDIR"
  --bind "$TMPBASE:$TMPBASE"
  --pwd "$WORKDIR"
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

# ------------------------------------------------------------
# Debug checks
# ------------------------------------------------------------
echo "[INFO] Debug: repo visibility in container..."
"${APPT_CMD[@]}" bash -lc "pwd; ls -lah '$WORKDIR' | head -n 30"
"${APPT_CMD[@]}" bash -lc "test -f '$WORKDIR/main.py' && echo '[OK] main.py exists' || (echo '[ERR] main.py missing' && exit 1)"

echo "[INFO] Debug: torch location..."
"${APPT_CMD[@]}" python - <<'PY'
import torch, sys
print("torch:", torch.__version__)
print("torch file:", torch.__file__)
print("sys.path head:", sys.path[:10])
PY

# ------------------------------------------------------------
# Run (Hydra overrides with ++ everywhere)
# ------------------------------------------------------------
echo "[INFO] Starting main.py ..."
"${APPT_CMD[@]}" bash -lc "
  cd '$WORKDIR'
  python '$WORKDIR/main.py' \
    hydra.run.dir='$RUN_DIR' \
    dataset=wikitext2_local \
    model=gpt2 \
    decoding=top_k \
    detector=modernbert_mage \
    data_selection=importance_sampling \
    num_iterations=10 \
    wandb_disabled=true \
    ++model.model_name_or_path='$GPT2_DIR' \
    ++model.tokenizer_name='$GPT2_DIR' \
    ++detector.model_path='$DET_DIR' \
    ++detector.tokenizer_name='$DET_DIR'
"

echo "[INFO] Done at $(date)"
echo "[INFO] Outputs: $RUN_DIR"
