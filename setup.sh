#!/usr/bin/bash

# project directory
export AUSSIE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# modules
ml purge
ml cuda/12.8 cudnn/11.6 anaconda/3.0

# conda
eval "$(conda shell.bash hook)"
conda activate deep

# auto completion
eval "$(python aussie.py -sc install=bash)"

# complete stacktrace
export HYDRA_FULL_ERROR=1

# resolve multithreading conflict
export MKL_THREADING_LAYER=GNU

export OMP_NUM_THREADS=32
export NUMEXPR_MAX_THREADS=32
