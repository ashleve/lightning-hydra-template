# Make sure you have installed the NVIDIA driver >= 361.93 and Docker >= 19.03
# https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(Native-GPU-Support)#prerequisites

ARG CUDA_VERSION=11.1

# build from official cuda image, devel version is needed for apex
FROM nvidia/cuda:${CUDA_VERSION}-devel

ENV CONDA_ENV_NAME=env
ENV PYTHON_VERSION=3.8
ENV PYTORCH_VERSION=1.8.1
ENV CUDA_TOOLKIT_VERSION=11.1

# Print environment variables
RUN printenv

# Create a working directory
RUN mkdir /workspace
WORKDIR /workspace

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Switch to bash shell
SHELL ["/bin/bash", "-c"]

# Install Miniconda and Python
ADD /miniconda3.sh /workspace
RUN bash miniconda3.sh -b -p /conda \
    && rm miniconda3.sh \
    && echo export PATH=/conda/bin:$PATH >> .bashrc
ENV PATH="/conda/bin:${PATH}"

# Create conda env
RUN conda create \
    -n ${CONDA_ENV_NAME} \
    python=${PYTHON_VERSION}

RUN source activate ${CONDA_ENV_NAME} \
    && conda install cudatoolkit=${CUDA_TOOLKIT_VERSION} pytorch=${PYTORCH_VERSION} torchvision torchaudio \
    -c pytorch -c conda-forge -y

# Copy the requirements file to the container 
# ADD /requirements.txt /workspace

# Install requirements
# RUN pip install -r requirements.txt

# Install Apex for mixed-precision training
# RUN git clone https://github.com/NVIDIA/apex \
#     && cd apex \
#     && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Activate conda env by default
CMD ["source activate ${CONDA_ENV_NAME}"]
