# Make sure you have installed the NVIDIA driver >= 361.93 and Docker >= 19.03
# https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(Native-GPU-Support)#prerequisites

ARG CUDA_VERSION=11.1


FROM nvidia/cuda:${CUDA_VERSION}-devel


ENV CONDA_ENV_NAME=env


# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    # ca-certificates \
    # sudo \
    # apt-utils \
    git \
    # bzip2 \
    # libx11-6 \
    wget \
    vim \
    && rm -rf /var/lib/apt/lists/*


# Create a working directory
RUN mkdir /workspace
WORKDIR /workspace


# Install Miniconda and Python
# ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda3.sh
ADD /miniconda3.sh /workspace
RUN /bin/bash miniconda3.sh -b -p /conda && rm miniconda3.sh \ 
    && echo export PATH=/conda/bin:$PATH >> .bashrc
ENV PATH="/conda/bin:${PATH}"


# Copy the requirements file to the container 
ADD /conda_env_gpu.yaml /workspace
ADD /requirements.txt /workspace


# switch to bash shell
SHELL ["/bin/bash", "-c"]


# Create conda env
RUN conda env create \
    -f conda_env_gpu.yaml \
    -n ${CONDA_ENV_NAME} \
    && source activate ${CONDA_ENV_NAME}


# Install PyTorch
# RUN conda install -y pytorch=${TORCH_VERSION} torchvision -c pytorch -c conda-forge


# Install Apex for mixed-precision training
RUN git clone https://github.com/NVIDIA/apex \
    && cd apex \
    && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./


# Install requirements
RUN pip install -r requirements.txt


ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility


# Set the default command
CMD ["source activate ${CONDA_ENV_NAME}"]


# sudo docker run --gpus all -it --rm bbb

