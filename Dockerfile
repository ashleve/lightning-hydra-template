# Build: sudo docker build -t <project_name> .
# Run: sudo docker run -v $(pwd):/workspace/project --gpus all -it --rm <project_name>


FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04


ENV CONDA_ENV_NAME=myenv
ENV PYTHON_VERSION=3.8


# Basic setup
RUN apt update
RUN apt install -y bash \
    build-essential \
    git \
    curl \
    ca-certificates \
    wget \
    && rm -rf /var/lib/apt/lists


# Set working directory
WORKDIR /workspace/project


# Install Miniconda and create main env
ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda3.sh
RUN /bin/bash miniconda3.sh -b -p /conda \
    && echo export PATH=/conda/bin:$PATH >> .bashrc \
    && rm miniconda3.sh
ENV PATH="/conda/bin:${PATH}"
RUN conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION}


# Switch to bash shell
SHELL ["/bin/bash", "-c"]


# Install requirements
COPY requirements.txt ./
RUN source activate ${CONDA_ENV_NAME} \
    && pip install --no-cache-dir -r requirements.txt \
    && rm requirements.txt


# Set ${CONDA_ENV_NAME} to default virutal environment
RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc
