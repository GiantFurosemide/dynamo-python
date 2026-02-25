#!/bin/bash
#SBATCH --job-name=pydynamo_crop
#SBATCH --output=slurm_crop_%j.log
#SBATCH --error=slurm_crop_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=cpu
#SBATCH --time=04:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/home/muwang/Documents/GitHub/dynamo-python}"
CONDA_EXE="${CONDA_EXE:-/fast/AG_Kudryashev/PatrickMuWang/opt/manba/miniforge3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-pydynamo}"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/config/crop.yaml}"

echo "============================================================"
echo "[pydynamo-crop] start"
echo "[pydynamo-crop] repo:   ${REPO_ROOT}"
echo "[pydynamo-crop] config: ${CONFIG_PATH}"
echo "============================================================"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[pydynamo-crop] ERROR: config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "[pydynamo-crop] ERROR: repo root not found: ${REPO_ROOT}" >&2
  exit 1
fi

# 1) Activate conda environment
eval "$("${CONDA_EXE}" shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${REPO_ROOT}"
python -m pydynamo --help >/dev/null
python -m pydynamo crop --i "${CONFIG_PATH}"

echo "[pydynamo-crop] done"
