# build from official nvidia pytorch image
# https://ngc.nvidia.com/catalog/containers/nvidia:pytorch
FROM nvcr.io/nvidia/pytorch:21.03-py3


# Install python packages
RUN pip install pytorch-lightning \
    && pip install hydra-core --pre \
    && pip install hydra-colorlog --pre \
    && pip install hydra-optuna-sweeper --pre \
    && pip install python-dotenv \
    && pip install pre-commit \
    && pip install rich \
    && pip install neptune-client \ 
    && pip install wandb
