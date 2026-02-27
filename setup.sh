#!/usr/bin/bash

# project directory
export AUSSIE_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# conda
eval "$(conda shell.bash hook)"
conda activate aussie

# auto completion
eval "$(python aussie.py -sc install=bash)"

# complete stacktrace
export HYDRA_FULL_ERROR=1

# resolve multithreading conflict
export MKL_THREADING_LAYER=GNU
