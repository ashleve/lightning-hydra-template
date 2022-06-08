# setup.py
#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="src",
    version="0.0.1",
    description="Describe Your Cool Project",
    author="",
    author_email="",
    url="https://github.com/YourSeed",  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=["pytorch-lightning"],
    packages=find_packages(),
)
