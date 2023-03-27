#!/usr/bin/env python

from setuptools import find_packages, setup

INSTALL_REQUIRES = [
    "dill>=0.3",
    "hydra-core",
    "matplotlib>=3.5",
    "numpy>=1.21",
    "pandas>=1.3",
    "pyarrow",
    "pydantic",
    "pytorch-lightning",
    "seaborn>=0.12.2",
    "scikit_learn>=1.0",
    "scipy>=1.7",
    "sqlalchemy>=1.4",
    "timer>=0.2",
    "tqdm>=4.64",
    "wandb>=0.12",
]

setup(
    name="src",
    version="0.0.1",
    description="Developing a encoder transformer arcitekture for",
    author="",
    author_email="",
    url="https://github.com/MoustHolmes/IceCubeEncoderTransformer", 
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
)
