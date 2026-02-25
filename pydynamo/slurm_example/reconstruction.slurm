#!/bin/bash
#SBATCH --job-name=pydynamo_recon
#SBATCH --output=slurm_recon_%j.log
#SBATCH --error=slurm_recon_%j.err
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
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/config/reconstruction.yaml}"

echo "============================================================"
echo "[pydynamo-reconstruction] start"
echo "[pydynamo-reconstruction] repo:   ${REPO_ROOT}"
echo "[pydynamo-reconstruction] config: ${CONFIG_PATH}"
echo "============================================================"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[pydynamo-reconstruction] ERROR: config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "[pydynamo-reconstruction] ERROR: repo root not found: ${REPO_ROOT}" >&2
  exit 1
fi

# 1) Activate conda environment
eval "$("${CONDA_EXE}" shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${REPO_ROOT}"
python -m pydynamo --help >/dev/null
python -m pydynamo reconstruction --i "${CONFIG_PATH}"

echo "[pydynamo-reconstruction] done"
