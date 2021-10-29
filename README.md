# Dockerfile

This Dockerfile is for GPU only.

Make sure you have installed the NVIDIA driver >= 361.93 and Docker >= 19.03. <br>
https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(Native-GPU-Support)#prerequisites

You will need to [install Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to enable GPU support. <br>

Copy the Dockerfile to the template root folder.

To build the container use:

```bash
docker build -t <project_name> .
```

To mount the project to the container use:

```bash
docker run -v $(pwd):/workspace/project --gpus all -it --rm <project_name>
```
