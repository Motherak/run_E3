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
# Node-local temp (avoid ~/.local leakage)
# ------------------------------------------------------------
TMPBASE="${SLURM_TMPDIR:-/tmp}/${USER}/model_collapse_${SLURM_JOB_ID}"
mkdir -p "$TMPBASE"

export HF_HOME="$TMPBASE/.hf_cache"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export HUGGINGFACE_HUB_CACHE="$HF_HOME/hub"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false

# --- CRITICAL: prevent host python packages leaking into container ---
export PYTHONNOUSERSITE=1
unset PYTHONPATH
unset PYTHONUSERBASE

# Make container HOME clean
export APPTAINERENV_HOME="$TMPBASE/home"
mkdir -p "$TMPBASE/home"

# ------------------------------------------------------------
# Enter repo
# ------------------------------------------------------------
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
# Model paths (INSIDE container)
# ------------------------------------------------------------
GPT2_DIR="/opt/models/gpt2"
DET_DIR="/opt/models/modernbert_ai_detection"

mkdir -p "$RUN_DIR"

echo "[INFO] RUN_DIR: $RUN_DIR"
echo "[INFO] Using container: $IMG"
echo "[INFO] TMPBASE: $TMPBASE"

# ------------------------------------------------------------
# Run (Hydra overrides with ++ everywhere)
# ------------------------------------------------------------
apptainer exec --cleanenv --nv --bind "$WORKDIR:$WORKDIR" "$IMG" \
  python main.py \
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
