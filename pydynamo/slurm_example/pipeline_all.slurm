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

CONDA_EXE="/fast/AG_Kudryashev/PatrickMuWang/opt/manba/miniforge3/bin/conda"
CONDA_ENV="pydynamo"
CROP_CFG="config/crop.yaml"
RECON_CFG="config/reconstruction.yaml"
ALIGN_CFG="config/alignment.yaml"
CLASS_CFG="config/classification.yaml"

# 1) Activate conda environment
eval "$("${CONDA_EXE}" shell.bash hook)"
conda activate "${CONDA_ENV}"

pydynamo --help >/dev/null
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

pydynamo crop --i "${CROP_CFG}"
pydynamo reconstruction --i "${RECON_CFG}"
pydynamo alignment --i "${ALIGN_CFG}"
pydynamo classification --i "${CLASS_CFG}"

echo "[pydynamo-pipeline] done"
