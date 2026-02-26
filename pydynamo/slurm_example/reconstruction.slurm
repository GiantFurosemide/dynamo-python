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

CONDA_EXE="/fast/AG_Kudryashev/PatrickMuWang/opt/manba/miniforge3/bin/conda"
CONDA_ENV="pydynamo"
CONFIG_PATH="config/reconstruction.yaml"

# 1) Activate conda environment
eval "$("${CONDA_EXE}" shell.bash hook)"
conda activate "${CONDA_ENV}"

pydynamo --help >/dev/null
pydynamo reconstruction --i "${CONFIG_PATH}"

echo "[pydynamo-reconstruction] done"
