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

# Iterations (repo expects RUN_DIR/<i>/data.json)
NUM_ITERS=10

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

# Sanity check dataset files
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
mkdir -p "$RUN_DIR"
for i in $(seq 0 $NUM_ITERS); do
  mkdir -p "$RUN_DIR/$i"
  cp -f "$WORKDIR/data/wikitext2/train.json" "$RUN_DIR/$i/data.json"
done
echo "[INFO] Precreated $((NUM_ITERS+1)) folders with data.json under: $RUN_DIR"

# ------------------------------------------------------------
# Apptainer isolation: prevent host ~/.local leaking into container
# ------------------------------------------------------------
APPT_CMD=(apptainer exec --nv --cleanenv --containall --no-home
  --bind "$WORKDIR:$WORKDIR"
  --bind "$TMPBASE:$TMPBASE"
  --pwd "$WORKDIR"
  --env APPTAINERENV_HOME="$TMPBASE/home"
  --env APPTAINERENV_PYTHONNOUSERSITE=1
  --env APPTAINERENV_TOKENIZERS_PARALLELISM=false

  # Offline everywhere
  --env APPTAINERENV_TRANSFORMERS_OFFLINE=1
  --env APPTAINERENV_HF_HUB_OFFLINE=1
  --env APPTAINERENV_HF_HOME="$HF_HOME"
  --env APPTAINERENV_TRANSFORMERS_CACHE="$TRANSFORMERS_CACHE"
  --env APPTAINERENV_HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
  --env APPTAINERENV_HUGGINGFACE_HUB_CACHE="$HUGGINGFACE_HUB_CACHE"

  # W&B hard disable
  --env APPTAINERENV_WANDB_MODE=disabled
  --env APPTAINERENV_WANDB_DISABLED=true
  --env APPTAINERENV_WANDB_SILENT=true

  "$IMG"
)

# ------------------------------------------------------------
# Debug checks
# ------------------------------------------------------------
echo "[INFO] Debug: repo visible in container?"
"${APPT_CMD[@]}" bash -lc "pwd; ls -lah '$WORKDIR' | head -n 40; test -f '$WORKDIR/main.py' && echo '[OK] main.py exists'"

echo "[INFO] Debug: torch + transformers inside container"
"${APPT_CMD[@]}" python - <<'PY'
import torch, transformers, certifi, requests, huggingface_hub
print("torch:", torch.__version__)
print("transformers:", transformers.__version__)
print("certifi:", certifi.__version__)
print("requests:", requests.__version__)
print("hub:", huggingface_hub.__version__)
PY

echo "[INFO] Debug: data.json exists?"
ls -lah "$RUN_DIR/0/data.json" "$RUN_DIR/$NUM_ITERS/data.json"

# ------------------------------------------------------------
# Run
# Models are inside container at /opt/models/*
# Hydra overrides use ++ to force override existing keys.
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
