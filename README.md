<div align="center">

# PyTorch Lightning + Hydra Template

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-orange?style=for-the-badge&logo=pytorch"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-blueviolet?style=for-the-badge"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra-blue?style=for-the-badge"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge"></a>

A clean and scalable template to kickstart your deep learning project 🚀⚡🔥<br>
Click on [<kbd>Use this template</kbd>](https://github.com/hobogalaxy/lightning-hydra-template/generate) to initialize new repository.

This template tries to be as general as possible.
You should be able to easily modify behavior in [train.py](train.py) in case you need some unconventional configuration wiring.

*This template is work in progress.* <br>
*Suggestions are always welcome!*

</div>
<br>
<br>


If you use this template please add <br>
[![](https://shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/hobogalaxy/lightning-hydra-template) <br>
to your `README.md`.
<br>


**Contents**
- [PyTorch Lightning + Hydra Template](#pytorch-lightning--hydra-template)
  - [Why Lightning + Hydra?](#why-lightning--hydra)
  - [Main Ideas](#main-ideas)
  - [Some Notes](#some-notes)
  - [Project Structure](#project-structure)
  - [Quickstart](#quickstart)
    - [Your Superpowers](#your-superpowers)
  - [Features](#features)
  - [Main Project Configuration](#main-project-configuration)
  - [Experiment Configuration](#experiment-configuration)
    - [Simple Example](#simple-example)
    - [Advanced Example](#advanced-example)
  - [Workflow](#workflow)
  - [Logs](#logs)
  - [Experiment Tracking](#experiment-tracking)
  - [Tests](#tests)
  - [Distributed Training](#distributed-training)
  - [Linting](#linting)
  - [Tricks](#tricks)
  - [Examples Of Repositories Using This Template](#examples-of-repositories-using-this-template)
  - [Installing project as a package](#installing-project-as-a-package)
<br>


## Why Lightning + Hydra?
- <b>[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning)</b> provides great abstractions for well structured ML code and advanced features like checkpointing, gradient accumulation, distributed training, etc.
- <b>[Hydra](https://github.com/facebookresearch/hydra)</b> provides convenient way to manage experiment configurations and advanced features like overriding any config parameter from command line, scheduling execution of many runs, etc.
<br>


## Main Ideas
- Predefined Structure: clean and scalable so that work can easily be extended and replicated (see [#Project Structure](#project-structure))
- Rapid Experimentation: thanks to automating pipeline with config files and hydra command line superpowers
- Little Boilerplate: so pipeline can be easily modified (see [train.py](train.py))
- Main Configuration: main config file specifies default training configuration (see [#Main Project Configuration](#main-project-configuration))
- Experiment Configurations: stored in a separate folder, they can be composed out of smaller configs, override chosen parameters or define everything from scratch (see [#Experiment Configuration](#experiment-configuration))
- Experiment Tracking: many logging frameworks can be easily integrated! (see [#Experiment Tracking](#experiment-tracking))
- Logs: all logs (checkpoints, data from loggers, chosen hparams, etc.) are stored in a convenient folder structure imposed by Hydra (see [#Logs](#logs))
- Smoke Tests: simple bash scripts running 1-2 epoch experiments to check if your model doesn't crash under different conditions (see [tests](tests/))
- Hyperparameter Search: made easier with Hydra built in plugins like [Optuna Sweeper](https://hydra.cc/docs/next/plugins/optuna_sweeper)
- Workflow: comes down to 4 simple steps (see [#Workflow](#workflow))
<br>


## Some Notes
- ***Warning: this template currently uses development version of hydra which might be unstable (we wait until Hydra 1.1 is released).*** <br>
- *Inspired by:
[PyTorchLightning/deep-learninig-project-template](https://github.com/PyTorchLightning/deep-learning-project-template),
[drivendata/cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science),
[tchaton/lightning-hydra-seed](https://github.com/tchaton/lightning-hydra-seed),
[Erlemar/pytorch_tempest](https://github.com/Erlemar/pytorch_tempest),
[ryul99/pytorch-project-template](https://github.com/ryul99/pytorch-project-template).*
- *Check out [lucmos/nn-template](https://github.com/lucmos/nn-template) for easier-to-start but less scalable version of this template.*
- *To learn how to configure PyTorch with Hydra take a look at [this detailed MNIST tutorial](https://github.com/pytorch/hydra-torch/blob/master/examples/mnist_00.md).*
- *Repositories useful for configuring PyTorch and PyTorch Lightning classes with Hydra:
[romesco/hydra-lightning](https://github.com/romesco/hydra-lightning),
[pytorch/hydra-torch](https://github.com/pytorch/hydra-torch).*
- *Suggestions are always welcome!*
<br>


## Project Structure
The directory structure of new project looks like this:
```
├── configs                 <- Hydra configuration files
│   ├── trainer                 <- Configurations of Lightning trainers
│   ├── model                   <- Configurations of Lightning models
│   ├── datamodule              <- Configurations of Lightning datamodules
│   ├── callbacks               <- Configurations of Lightning callbacks
│   ├── logger                  <- Configurations of Lightning loggers
│   ├── experiment              <- Configurations of experiments
│   │
│   ├── config.yaml             <- Main project configuration file
│   └── config_optuna.yaml      <- Configuration of Optuna hyperparameter search
│
├── data                    <- Project data
│
├── logs                    <- Logs generated by Hydra and PyTorch Lightning loggers
│
├── notebooks               <- Jupyter notebooks
│
├── tests                   <- Tests of any kind
│   ├── smoke_tests.sh          <- A couple of quick experiments to test if your model
│   │                              doesn't crash under different training conditions
│   └── ...
│
├── src
│   ├── architectures           <- PyTorch model architectures
│   ├── callbacks               <- PyTorch Lightning callbacks
│   ├── datamodules             <- PyTorch Lightning datamodules
│   ├── datasets                <- PyTorch datasets
│   ├── models                  <- PyTorch Lightning models
│   ├── transforms              <- Data transformations
│   └── utils                   <- Utility scripts
│       ├── inference_example.py    <- Example of inference with trained model
│       └── template_utils.py       <- Some extra template utilities
│
├── train.py                <- Train model with chosen experiment configuration
│
├── .gitignore
├── .pre-commit-config.yaml <- Configuration of hooks for automatic code formatting
├── LICENSE
├── README.md
├── conda_env_gpu.yaml      <- File for installing conda env for GPU
├── conda_env_cpu.yaml      <- File for installing conda env for CPU
├── requirements.txt        <- File for installing python dependencies
└── setup.py                <- File for installing project as a package
```
<br>


## Quickstart
```yaml
# clone project
git clone https://github.com/hobogalaxy/lightning-hydra-template
cd lightning-hydra-template

# [OPTIONAL] create conda environment
conda env create -f conda_env_gpu.yaml -n testenv
conda activate testenv

# install requirements
pip install -r requirements.txt
```

When running `python train.py` you should see something like this:
<div align="center">

![](https://github.com/hobogalaxy/lightning-hydra-template/blob/resources/teminal.png)

</div>

### Your Superpowers
(click to expand)

<details>
<summary>Override any config parameter from command line</summary>

```yaml
python train.py trainer.max_epochs=20 model.lr=0.0005
```

</details>


<details>
<summary>Train on GPU</summary>

```yaml
python train.py trainer.gpus=1
```

</details>


<details>
  <summary>Train model with any logger available in PyTorch Lightning, like <a href="https://wandb.ai/">Weights&Biases</a></summary>

```yaml
# set project and entity names in `configs/logger/wandb.yaml`
wandb:
    project: "your_project_name"
    entity: "your_wandb_team_name"
```

```yaml
# train model with Weights&Biases
python train.py logger=wandb
```

</details>


<details>
<summary>Train model with chosen experiment config</summary>

```yaml
# experiment configurations are placed in folder `configs/experiment/`
python train.py +experiment=exp_example_simple
```

</details>


<details>
<summary>Execute all experiments from folder</summary>

```yaml
# execute all experiments from folder `configs/experiment/`
python train.py -m '+experiment=glob(*)'
```

</details>


<details>
<summary>Attach some callbacks to run</summary>

```yaml
# callback set configurations are placed in `configs/callbacks/`
python train.py callbacks=default_callbacks
```

</details>


<details>
<summary>Easily debug</summary>

```yaml
# run 1 train, val and test loop, using only 1 batch
python train.py debug=True
```

</details>


<details>
<summary>Resume training from checkpoint</summary>

```yaml
# checkpoint can be either path or URL
# path should be absolute!
python train.py trainer.resume_from_checkpoint="/home/user/X/lightning-hydra-template/logs/runs/2021-02-28/16-50-49/checkpoints/last.ckpt"
# currently loading ckpt in Lightning doesn't resume logger experiment, this should change when v1.3 is released...
```

</details>


<details>
<summary>Create a sweep over some hyperparameters </summary>

```yaml
# this will run 6 experiments one after the other,
# each with different combination of batch_size and learning rate
python train.py -m datamodule.batch_size=32,64,128 model.lr=0.001,0.0005
```

</details>


<details>
<summary>Create a sweep over some hyperparameters with Optuna</summary>

```yaml
# this will run hyperparameter search defined in `configs/config_optuna.yaml`
# over chosen experiment config
python train.py -m --config-name config_optuna.yaml +experiment=exp_example_simple
```

</details>

<br>


## Features
- Method to pretty print configuration composed by Hydra at the start of the run, using [Rich](https://github.com/willmcgugan/rich/) library ([template_utils.py](src/utils/template_utils.py))
- Method to log chosen parts of Hydra config to all loggers ([template_utils.py](src/utils/template_utils.py))
- Example of hyperparameter search with Optuna sweeps ([config_optuna.yaml](configs/config_optuna.yaml))
- ~~Example of hyperparameter search with Weights&Biases sweeps~~ (TODO)
- Examples of simple bash scripts to check if your model doesn't crash under different training conditions ([tests/](tests/))
- Example of inference with trained model  ([inference_example.py](src/utils/inference_example.py))
- Built in requirements ([requirements.txt](requirements.txt))
- Built in conda environment initialization ([conda_env_gpu.yaml](conda_env_gpu.yaml), [conda_env_cpu.yaml](conda_env_cpu.yaml))
- Built in python package setup ([setup.py](setup.py))
- Built in pre-commit hooks for automatic code formatting ([pre-commit-config.yaml](pre-commit-config.yaml))
- Example with MNIST classification ([mnist_model.py](src/models/mnist_model.py), [mnist_datamodule.py](src/datamodules/mnist_datamodule.py))
- Optional callbacks for Weigths&Biases ([wandb_callbacks.py](src/callbacks/wandb_callbacks.py))
  - To support reproducibility:
    - UploadCodeToWandbAsArtifact
    - UploadCheckpointsToWandbAsArtifact
    - WatchModelWithWandb
  - To provide examples of logging custom visualisations and metrics with callbacks:
    - LogBestMetricScoresToWandb
    - LogF1PrecisionRecallHeatmapToWandb
    - LogConfusionMatrixToWandb
<br>


## Main Project Configuration
Location: [configs/config.yaml](configs/config.yaml)<br>
Main project config contains default training configuration.<br>
It determines how config is composed when simply executing command: `python train.py`
```yaml
# specify here default training configuration
defaults:
    - trainer: default_trainer.yaml
    - model: mnist_model.yaml
    - datamodule: mnist_datamodule.yaml
    - callbacks: default_callbacks.yaml  # set this to null if you don't want to use callbacks
    - logger: null  # set logger here or use command line (e.g. `python train.py logger=wandb`)


# path to original working directory (that `train.py` was executed from in command line)
# hydra hijacks working directory by changing it to the current log directory,
# so it's useful to have path to original working directory as a special variable
# read more here: https://hydra.cc/docs/next/tutorials/basic/running_your_app/working_directory
work_dir: ${hydra:runtime.cwd}


# path to folder with data
data_dir: ${work_dir}/data/


# use `python train.py debug=true` for easy debugging!
# (equivalent to running `python train.py trainer.fast_dev_run=True`)
debug: False


# pretty print config at the start of the run using Rich library
print_config: True


# disable python warnings if they annoy you
disable_warnings: False


# disable lightning logs if they annoy you
disable_lightning_logs: False


# hydra configuration
hydra:
    
    # output paths for hydra logs
    run:
        dir: logs/runs/${now:%Y-%m-%d}/${now:%H-%M-%S}
    sweep:
        dir: logs/multiruns/${now:%Y-%m-%d_%H-%M-%S}
        subdir: ${hydra.job.num}
    
    # set your environment variables here
    job:
        env_set:
            ENV_VAR_X: something
```
<br>


## Experiment Configuration
Location: [configs/experiment](configs/experiment)<br>
You should store all your experiment configurations in this folder.<br>
### Simple Example
```yaml
# to execute this experiment run:
# python train.py +experiment=exp_example_simple

defaults:
    - override /trainer: default_trainer.yaml
    - override /model: mnist_model.yaml
    - override /datamodule: mnist_datamodule.yaml
    - override /callbacks: default_callbacks.yaml
    - override /logger: null

# all parameters below will be merged with parameters from default configurations set above
# this allows you to overwrite only specified parameters

seed: 12345

trainer:
    max_epochs: 10
    gradient_clip_val: 0.5

model:
    lr: 0.001
    lin1_size: 128
    lin2_size: 256
    lin3_size: 64

datamodule:
    batch_size: 64
    train_val_test_split: [55_000, 5_000, 10_000]
```
<br>

### Advanced Example
```yaml
# to execute this experiment run:
# python train.py +experiment=exp_example_full

defaults:
    - override /trainer: null
    - override /model: null
    - override /datamodule: null
    - override /callbacks: null
    - override /logger: null

# we override default configurations with nulls to prevent them from loading at all
# instead we define all modules and their paths directly in this config,
# so everything is stored in one place for more readibility

seed: 12345

trainer:
    _target_: pytorch_lightning.Trainer
    gpus: 0
    min_epochs: 1
    max_epochs: 10
    gradient_clip_val: 0.5

model:
    _target_: src.models.mnist_model.LitModelMNIST
    optimizer: adam
    lr: 0.001
    weight_decay: 0.00005
    architecture: SimpleDenseNet
    input_size: 784
    lin1_size: 256
    dropout1: 0.30
    lin2_size: 256
    dropout2: 0.25
    lin3_size: 128
    dropout3: 0.20
    output_size: 10

datamodule:
    _target_: src.datamodules.mnist_datamodule.MNISTDataModule
    data_dir: ${data_dir}
    batch_size: 64
    train_val_test_split: [55_000, 5_000, 10_000]
    num_workers: 0
    pin_memory: False

logger:
    wandb:
        tags: ["best_model", "uwu"]
        notes: "Description of this model."
```
<br>


## Workflow
1. Write your PyTorch Lightning model (see [mnist_model.py](src/models/mnist_model.py) for example)
2. Write your PyTorch Lightning datamodule (see [mnist_datamodule.py](src/datamodules/mnist_datamodule.py) for example)
3. Write your experiment config, containing paths to your model and datamodule (see [configs/experiment](configs/experiment) for examples)
4. Run training with chosen experiment config:<br>
    ```bash
    python train.py +experiment=experiment_name.yaml
    ```
<br>


## Logs
Hydra creates new working directory for every executed run. <br>
By default, logs have the following structure:
```
│
├── logs
│   ├── runs                    # Folder for logs generated from single runs
│   │   ├── 2021-02-15              # Date of executing run
│   │   │   ├── 16-50-49                # Hour of executing run
│   │   │   │   ├── .hydra                  # Hydra logs
│   │   │   │   ├── wandb                   # Weights&Biases logs
│   │   │   │   ├── checkpoints             # Training checkpoints
│   │   │   │   └── ...                     # Any other thing saved during training
│   │   │   ├── ...
│   │   │   └── ...
│   │   ├── ...
│   │   └── ...
│   │
│   └── multiruns               # Folder for logs generated from multiruns (sweeps)
│       ├── 2021-02-15_16-50-49     # Date and hour of executing sweep
│       │   ├── 0                       # Job number
│       │   │   ├── .hydra                  # Hydra logs
│       │   │   ├── wandb                   # Weights&Biases logs
│       │   │   ├── checkpoints             # Training checkpoints
│       │   │   └── ...                     # Any other thing saved during training
│       │   ├── 1
│       │   ├── 2
│       │   └── ...
│       ├── ...
│       └── ...
│
```
You can change this structure by modifying paths in [config.yaml](configs/config.yaml).
<br><br>


## Experiment Tracking
PyTorch Lightning provides built in loggers for Weights&Biases, Neptune, Comet, MLFlow, Tensorboard and CSV. To use one of them, simply add its config to [configs/logger](configs/logger) and run:
 ```
 python train.py logger=logger_config.yaml
 ```
You can use many of them at once (see [configs/logger/many_loggers.yaml](configs/logger/many_loggers.yaml) for example).
<br><br>


## Tests
(TODO) <br>
To execute:
```bash
bash tests/smoke_tests.sh
```
<br><br>


## Distributed Training
(TODO)
<br><br>


## Linting
(TODO)
<br><br>


## Tricks
(TODO)
<!-- installing miniconda, PrettyErrors and Rich exception handling, VSCode setup,
k-fold cross validation, linter, faster tab completion import trick,
choosing metric names with '/' for wandb -->
<br><br>


## Examples Of Repositories Using This Template
(TODO)
<br><br>



<br>
<br>
<br>

### DELETE EVERYTHING ABOVE FOR YOUR PROJECT

---

<div align="center">

# Your Project Name

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-orange?logo=pytorch"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-blueviolet"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-blue"></a>
[![](https://shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/hobogalaxy/lightning-hydra-template)

</div>

## Description
What it does

## How to run
First, install dependencies:
```yaml
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# optionally create conda environment
conda env create -f conda_env_gpu.yaml -n your_env_name
conda activate your_env_name

# install requirements
pip install -r requirements.txt
```

Next, you can train model with default configuration without logging:
```yaml
python train.py
```

Or you can train model with chosen logger like Weights&Biases:
```yaml
# set project and entity names in `configs/logger/wandb.yaml`
wandb:
    project: "your_project_name"
    entity: "your_wandb_team_name"
```

```yaml
# train model with Weights&Biases
python train.py logger=wandb
```

Train model with chosen experiment config:
```yaml
# experiment configurations are placed in folder `configs/experiment/`
python train.py +experiment=exp_example_simple
```

You can override any parameter from command line like this:
```yaml
python train.py trainer.max_epochs=20 model.lr=0.0005
```

To train on GPU:
```yaml
python train.py trainer.gpus=1
```

<br>


## Installing project as a package
Optionally you can install project as a package with [setup.py](setup.py):
```yaml
# install from local files
pip install -e .

# or install from git repo
pip install git+git://github.com/YourGithubName/your-repo-name.git --upgrade
```
So you can easily import any file into any other file like so:
```python
from src.models.mnist_model import LitModelMNIST
from src.datamodules.mnist_datamodule import MNISTDataModule
```
