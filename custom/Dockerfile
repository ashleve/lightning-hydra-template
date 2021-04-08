# Make sure you have installed the NVIDIA driver >= 361.93 and Docker >= 19.03
# https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(Native-GPU-Support)#prerequisites

ARG CUDA_VERSION=11.1

# build from official cuda image, devel version is needed for apex
# https://ngc.nvidia.com/catalog/containers/nvidia:cuda
FROM nvidia/cuda:${CUDA_VERSION}-devel


ENV CONDA_ENV_NAME=myenv
ENV PYTHON_VERSION=3.8
ENV PYTORCH_VERSION=1.8.1
ENV CUDA_TOOLKIT_VERSION=11.1


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
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda3.sh
RUN /bin/bash miniconda3.sh -b -p /conda \
    && rm miniconda3.sh \
    && echo export PATH=/conda/bin:$PATH >> .bashrc
ENV PATH="/conda/bin:${PATH}"


# Create conda env
RUN conda create \
    -n ${CONDA_ENV_NAME} \
    python=${PYTHON_VERSION}


# Install PyTorch
RUN source activate ${CONDA_ENV_NAME} \
    && conda install cudatoolkit=${CUDA_TOOLKIT_VERSION} pytorch=${PYTORCH_VERSION} torchvision torchaudio \
    -c pytorch -c conda-forge -y


# Install Apex for mixed-precision training
RUN source activate ${CONDA_ENV_NAME} \
    && git clone https://github.com/NVIDIA/apex \
    && cd apex \
    && pip install -v --disable-pip-version-check --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./ \
    && cd .. \
    && rm -r apex


# Install python packages
RUN source activate ${CONDA_ENV_NAME} \
    && pip install \
    pytorch-lightning \
    hydra-core>=1.1.0.dev5 \
    hydra-colorlog \
    hydra-optuna-sweeper \
    python-dotenv \
    pre-commit \
    rich \
    neptune wandb mlflow


# Download template
RUN source activate ${CONDA_ENV_NAME} \
    && git clone https://github.com/ashleve/lightning-hydra-template \
    && cd lightning-hydra-template \
    && pre-commit install


# Set ${CONDA_ENV_NAME} to default virutal environment
RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc
