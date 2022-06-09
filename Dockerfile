# Build: sudo docker build -t <project_name> .
# Run: sudo docker run -v $(pwd):/workspace/project --gpus all -it --rm <project_name>

ARG PYTORCH="1.11.0"
ARG CUDA="11.5"
ARG CUDNN="8"
ARG UBUNTU="22.04"

FROM nvcr.io/nvidia/pytorch:22.05-py3

# Basic setup
RUN apt update

# Set working directory
WORKDIR /workspace/project

# Install Miniconda and create main env
# ADD https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh miniconda3.sh
# RUN /bin/bash miniconda3.sh -b -p /conda \
#     && echo export PATH=/conda/bin:$PATH >> .bashrc \
#     && rm miniconda3.sh
# ENV PATH="/conda/bin:${PATH}"
# RUN conda create -n ${CONDA_ENV_NAME} python=${PYTHON_VERSION}

# Switch to bash shell
# SHELL ["/bin/bash", "-c"]

# Install requirements
COPY requirements.txt ./

# RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113

# RUN source activate ${CONDA_ENV_NAME} \
#    && pip3 install --no-cache-dir -r requirements.txt \
#    && rm requirements.txt

RUN pip3 install --no-cache-dir -r requirements.txt && rm requirements.txt
# RUN mkdir -p data/UDIVA

# Set ${CONDA_ENV_NAME} to default virutal environment
# RUN echo "source activate ${CONDA_ENV_NAME}" >> ~/.bashrc
