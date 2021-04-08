# build from official nvidia pytorch image
# https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:21.03-py3


RUN pip install \
    pytorch-lightning \
    hydra-core>=1.1.0.dev5 \
    hydra-colorlog \
    hydra-optuna-sweeper \
    python-dotenv \
    pre-commit \
    rich
