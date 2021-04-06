# image based on:
# https://ngc.nvidia.com/catalog/containers/nvidia:pytorch


# build from official nvidia pytorch image,
# contains:
# - conda (virtual environments)
# - apex (mixed precision training)
FROM nvcr.io/nvidia/pytorch:21.03-py3


RUN pip install \
    && pytorch-lightning>=1.2.5 \
    && hydra-core>=1.1.0.dev5 \
    && hydra-colorlog>=1.0.0 \
    && hydra-optuna-sweeper \
    && python-dotenv \ 
    && pre-commit \
    && rich


# Install .autoenv for automatic conda env activation and tab completion
RUN echo "autoenv() { [[ -f \"\$PWD/.autoenv\" ]] && source .autoenv ; } ; cd() { builtin cd \"\$@\" ; autoenv ; } ; autoenv" >> ~/.bashrc
