#!/bin/bash
# One-command runner for pydynamo
# Avoid OMP Error #179 (SHM2) in sandboxed/restricted envs
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

set -e
cd "$(dirname "$0")"
python -m pydynamo "$@"
