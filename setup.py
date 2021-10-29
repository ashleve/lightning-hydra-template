from setuptools import find_packages, setup

# you should change the name of "src" folder to your project name
# to support installing project as a package

# after that, project can be installed with `pip install -e .`
# or with `pip install git+git://github.com/YourGithubName/your-repo-name.git --upgrade`


setup(
    name="src",  # change "src" folder name to your project name
    version="0.0.0",
    description="Describe Your Cool Project",
    author="...",
    author_email="...",
    url="https://github.com/ashleve/lightning-hydra-template",  # replace with your own github project link
    install_requires=[
        "pytorch>=1.10.0",
        "pytorch-lightning>=1.4.0",
        "hydra-core==1.1.0",
    ],
    packages=find_packages(),
)
