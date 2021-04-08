# Dockerfiles

- **simple** - build from official nvidia pytorch image, contains: apex, miniconda
- **custom** - build from official nvidia cuda image, contains: apex, initialized miniconda environment, ready to run template, allows for specifying dependiecies (versions of cuda, python, pytorch)
<br>


> Make sure you have installed the NVIDIA driver >= 361.93 and Docker >= 19.03 <br>
https://github.com/NVIDIA/nvidia-docker/wiki/Installation-(Native-GPU-Support)#prerequisites
<br>


```bash
cd simple
docker build -t lightning-hydra .
docker run --gpus all -it --rm -v /home/USER/Desktop:/workspace/Desktop lightning-hydra
```
