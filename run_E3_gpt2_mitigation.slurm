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
IMG="$WORKDIR/model_collapse_tfdev.sif"
RUN_DIR="$WORKDIR/outputs_fixed/gpt2_topk_importance/job_${SLURM_JOB_ID}"

NUM_ITERS=10

# ------------------------------------------------------------
# Sanity checks
# ------------------------------------------------------------
if [ ! -d "$WORKDIR" ]; then
  echo "[ERROR] WORKDIR not found: $WORKDIR"
  exit 1
fi
if [ ! -f "$IMG" ]; then
  echo "[ERROR] Image not found: $IMG"
  exit 1
fi

cd "$WORKDIR"
echo "[INFO] Working dir: $(pwd)"
echo "[INFO] RUN_DIR: $RUN_DIR"
echo "[INFO] IMG: $IMG"

# ------------------------------------------------------------
# Node-local scratch (keep caches off /home; avoid container /tmp)
# ------------------------------------------------------------
TMPBASE="${SLURM_TMPDIR:-/tmp}/${USER}/model_collapse_${SLURM_JOB_ID}"
mkdir -p "$TMPBASE"

# we bind TMPBASE into container as /scratch
SCRATCH_IN_CONT="/scratch"
HF_HOME_CONT="$SCRATCH_IN_CONT/.hf_cache"

mkdir -p "$RUN_DIR"

echo "[INFO] TMPBASE(host): $TMPBASE"
echo "[INFO] HF_HOME(container): $HF_HOME_CONT"

# ------------------------------------------------------------
# Dataset config for Hydra + sanity
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
# Precreate expected generation folders
# ------------------------------------------------------------
for i in $(seq 0 $NUM_ITERS); do
  mkdir -p "$RUN_DIR/$i"
  cp -f "$WORKDIR/data/wikitext2/train.json" "$RUN_DIR/$i/data.json"
done
echo "[INFO] Precreated $((NUM_ITERS+1)) folders with data.json under: $RUN_DIR"

# ------------------------------------------------------------
# Apptainer command (clean, isolated, caches forced to /scratch)
# ------------------------------------------------------------
APPT_CMD=(apptainer exec --nv --cleanenv --containall --no-home
  --bind "$WORKDIR:$WORKDIR"
  --bind "$TMPBASE:$SCRATCH_IN_CONT"
  --pwd "$WORKDIR"

  # isolate HOME + avoid host ~/.local leakage
  --env HOME="$SCRATCH_IN_CONT/home"
  --env PYTHONNOUSERSITE=1
  --env TOKENIZERS_PARALLELISM=false

  # OFFLINE everywhere
  --env TRANSFORMERS_OFFLINE=1
  --env HF_HUB_OFFLINE=1

  # Force caches to node-local scratch (NOT /tmp/.hf_cache from %environment)
  --env HF_HOME="$HF_HOME_CONT"
  --env HF_DATASETS_CACHE="$HF_HOME_CONT/datasets"
  --env HUGGINGFACE_HUB_CACHE="$HF_HOME_CONT/hub"
  --env TRANSFORMERS_CACHE="$HF_HOME_CONT/transformers"

  # W&B hard disable
  --env WANDB_MODE=disabled
  --env WANDB_DISABLED=true
  --env WANDB_SILENT=true

  "$IMG"
)

# ------------------------------------------------------------
# Debug checks (short)
# ------------------------------------------------------------
echo "[INFO] Debug: main.py visible in container?"
"${APPT_CMD[@]}" bash -lc "test -f '$WORKDIR/main.py' && echo '[OK] main.py exists' || (echo '[ERROR] main.py missing' && exit 2)"

echo "[INFO] Debug: HF cache location inside container:"
"${APPT_CMD[@]}" bash -lc "python - <<'PY'
import os
print('HF_HOME=', os.environ.get('HF_HOME'))
print('HF_DATASETS_CACHE=', os.environ.get('HF_DATASETS_CACHE'))
PY"

# ------------------------------------------------------------
# Run
# Notes:
# - models are inside container at /opt/models/*
# - we only override detector paths (those keys exist in your config)
# - DO NOT pass model.model_name_or_path unless your Hydra config supports it
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
    num_iterations=$NUM_ITERS \
    wandb_disabled=true \
    ++detector.model_path='/opt/models/modernbert_ai_detection' \
    ++detector.tokenizer_name='/opt/models/modernbert_ai_detection'
"

echo "[INFO] Done at $(date)"
echo "[INFO] Outputs: $RUN_DIR"
