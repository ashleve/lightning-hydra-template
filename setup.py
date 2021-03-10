#!/usr/bin/env python
from setuptools import find_packages, setup

setup(
    name="src",
    version="0.0.0",
    description="Describe Your Cool Project",
    author="",
    author_email="",
    url="https://github.com/hobogalaxy/lightning-hydra-template",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=["pytorch-lightning>=1.2.1", "hydra-core>=1.0.6"],
    packages=find_packages(),
)
