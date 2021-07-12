#!/bin/bash
# Run from root folder with: bash bash/setup.sh

# Configure env
read -rp "Enter environment name: " env_name
read -rp "Enter python version (at least '3.7') " python_version
read -rp "Enter cuda version (e.g. '10.2', '11.1'  or 'none' to avoid installing cuda support): " cuda_version

# Create conda env
conda create -yn "$env_name" python="$python_version"
conda activate "$env_name"

# Install pytorch
if [ "$cuda_version" == "none" ]; then
    conda install -y pytorch torchvision cpuonly -c pytorch
else
    conda install -y pytorch torchvision cudatoolkit=$cuda_version -c pytorch
fi

# Install python requirements
pip install -r requirements.txt
