#!/usr/bin/env python
from setuptools import setup, find_packages


setup(
    name="src",
    version="0.0.0",
    description="Describe Your Cool Project",
    author="",
    author_email="",
    url="https://github.com/hobogalaxy/lightning-hydra-template",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=["pytorch-lightning", "hydra-core"],
    packages=find_packages(),
)
