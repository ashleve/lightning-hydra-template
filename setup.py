#!/usr/bin/env python
from setuptools import setup, find_packages


setup(
    name='project',
    version='0.0.0',
    description='Describe Your Cool Project',
    author='',
    author_email='',
    url='https://github.com/kinoai/deep-learning-project-template',  # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    install_requires=['pytorch-lightning', 'PyYAML'],
    packages=find_packages(),
)
