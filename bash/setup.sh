#!/bin/bash
# basic setup of virtual environment
# Run from root folder with: bash bash/setup.sh

# Setup conda
# source ~/miniconda3/etc/profile.d/conda.sh

# Create conda env
read -rp "Enter environment name: " env_name
read -rp "Enter python version (at least '3.7') " python_version
conda create -yn "$env_name" python="$python_version"
conda activate "$env_name"

# Install pytorch
read -rp "Enter cuda version (e.g. '10.2', '11.1'  or 'none' to avoid installing cuda support): " cuda_version
if [ "$cuda_version" == "none" ]; then
    conda install -y pytorch torchvision cpuonly -c pytorch
else
    conda install -y pytorch torchvision cudatoolkit=$cuda_version -c pytorch
fi

# Install python requirements
pip install -r requirements.txt
