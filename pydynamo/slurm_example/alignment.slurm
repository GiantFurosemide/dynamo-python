#!/bin/bash
#SBATCH --job-name=pydynamo_align
#SBATCH --output=slurm_align_%j.log
#SBATCH --error=slurm_align_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=08:00:00

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-/home/muwang/Documents/GitHub/dynamo-python}"
CONDA_EXE="${CONDA_EXE:-/fast/AG_Kudryashev/PatrickMuWang/opt/manba/miniforge3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-pydynamo}"
CONFIG_PATH="${CONFIG_PATH:-${SCRIPT_DIR}/config/alignment.yaml}"

echo "============================================================"
echo "[pydynamo-alignment] start"
echo "[pydynamo-alignment] repo:   ${REPO_ROOT}"
echo "[pydynamo-alignment] config: ${CONFIG_PATH}"
echo "============================================================"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[pydynamo-alignment] ERROR: config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

if [[ ! -d "${REPO_ROOT}" ]]; then
  echo "[pydynamo-alignment] ERROR: repo root not found: ${REPO_ROOT}" >&2
  exit 1
fi

# 1) Activate conda environment
eval "$("${CONDA_EXE}" shell.bash hook)"
conda activate "${CONDA_ENV}"

cd "${REPO_ROOT}"
python -m pydynamo --help >/dev/null
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
python -m pydynamo alignment --i "${CONFIG_PATH}"

echo "[pydynamo-alignment] done"
