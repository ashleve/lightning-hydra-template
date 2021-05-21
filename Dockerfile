# Build: docker build -t project_name .
# Run: docker run --gpus all -it --rm project_name

# Build from official Nvidia PyTorch image
# GPU-ready with Apex for mixed-precision support
# https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
# https://docs.nvidia.com/deeplearning/frameworks/support-matrix/
FROM nvcr.io/nvidia/pytorch:21.03-py3


# Copy all files
ADD . /workspace/project
WORKDIR /workspace/project


# Create myenv
RUN conda env create -f conda_env_gpu.yaml -n myenv
RUN conda init bash


# Set myenv to default virtual environment
RUN echo "source activate myenv" >> ~/.bashrc
