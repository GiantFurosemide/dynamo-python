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

CONDA_EXE="/fast/AG_Kudryashev/PatrickMuWang/opt/manba/miniforge3/bin/conda"
CONDA_ENV="pydynamo"
CONFIG_PATH="config/alignment.yaml"

# 1) Activate conda environment
eval "$("${CONDA_EXE}" shell.bash hook)"
conda activate "${CONDA_ENV}"

pydynamo --help >/dev/null
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"
pydynamo alignment --i "${CONFIG_PATH}"

echo "[pydynamo-alignment] done"
