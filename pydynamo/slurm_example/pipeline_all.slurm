#!/bin/bash
#SBATCH --job-name=pydynamo_pipeline
#SBATCH --output=slurm_pipeline_%j.log
#SBATCH --error=slurm_pipeline_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/home/muwang/Documents/GitHub/dynamo-python}"
CONDA_EXE="${CONDA_EXE:-/fast/AG_Kudryashev/PatrickMuWang/opt/manba/miniforge3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-pydynamo}"

CROP_CFG="${CROP_CFG:-${SCRIPT_DIR}/config/crop.yaml}"
RECON_CFG="${RECON_CFG:-${SCRIPT_DIR}/config/reconstruction.yaml}"
ALIGN_CFG="${ALIGN_CFG:-${SCRIPT_DIR}/config/alignment.yaml}"
CLASS_CFG="${CLASS_CFG:-${SCRIPT_DIR}/config/classification.yaml}"

run_step() {
  local name="$1"
  local cfg="$2"
  echo "============================================================"
  echo "[pydynamo-pipeline] step: ${name}"
  echo "[pydynamo-pipeline] config: ${cfg}"
  echo "============================================================"
  if [[ ! -f "${cfg}" ]]; then
    echo "[pydynamo-pipeline] ERROR: missing config for ${name}: ${cfg}" >&2
    exit 1
  fi
  python -m pydynamo "${name}" --i "${cfg}"
}

echo "============================================================"
echo "[pydynamo-pipeline] start"
echo "[pydynamo-pipeline] repo: ${REPO_ROOT}"
echo "============================================================"

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "[pydynamo-pipeline] ERROR: repo root not found: ${REPO_ROOT}" >&2
  exit 1
fi

# 1) Activate conda environment
eval "$("${CONDA_EXE}" shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${REPO_ROOT}"
python -m pydynamo --help >/dev/null
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

run_step crop "${CROP_CFG}"
run_step reconstruction "${RECON_CFG}"
run_step alignment "${ALIGN_CFG}"
run_step classification "${CLASS_CFG}"

echo "[pydynamo-pipeline] done"
