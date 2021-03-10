<div align="center">

# Lightning-Hydra-Template

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-orange?style=for-the-badge&logo=pytorch"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-blueviolet?style=for-the-badge"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra-blue?style=for-the-badge"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge"></a>

A clean and scalable template to kickstart your deep learning project 🚀⚡🔥<br>
Click on [<kbd>Use this template</kbd>](https://github.com/hobogalaxy/lightning-hydra-template/generate) to initialize new repository.

*This template is work in progress. Suggestions are always welcome!*

</div>


<br>


If you use this template please add <br>
[![](https://shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/hobogalaxy/lightning-hydra-template) <br>
to your `README.md`.
<br>


## Introduction
This template tries to be as general as possible.
By using it, you avoid writing any boilerplate code. At the same time it's flexible - you can easily delete any unwanted features from the pipeline or rewire the configuration, by modifying behavior in [train.py](train.py).

> Effective usage of this template requires learning of a couple of technologies: [PyTorch](https://pytorch.org), [PyTorch Lightning](https://www.pytorchlightning.ai) and [Hydra](https://hydra.cc). Knowledge of some experiment logging framework like [Weights&Biases](https://wandb.com), [Neptune](https://neptune.ai) or [MLFlow](https://mlflow.org) is also recommended.

The main advantage of using it, is that it allows you to rapidly iterate over new models and scale your projects from small single experiments to large hyperparameter searches on computing clusters, without writing any boilerplate code. To my knowledge, it might be the most convenient and all-in-one technology stack for Deep Learning research. It's also a collection of best practices for efficient workflow and reproducibility.

The main arguments for not using this template, are that Lightning and Hydra are not yet mature, which means you will probably run into some bugs. Also Lightning is not well suited for everything, e.g. for Reinforcement Learning it's probably better to replace it with Ray/RLlib.

### Why PyTorch Lightning?
PyTorch Lightning is a lightweight PyTorch wrapper for high-performance AI research.
Makes your code neatly organized and provides lots of useful features, like ability to run model on CPU, GPU, multi-GPU cluster and TPU.


### Why Hydra?
Hydra is an open-source Python framework that simplifies the development of research and other complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line. It provides convenient ways to manage experiments and advanced features like overriding any config parameter from command line or sweeping over hyperparameters.
<br>
<br>
<br>


## Main Ideas Of This Template
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
- Best Practices: a couple of recommended tools and standards (see [#Best Practices](#best-practices))
<br>


## Project Structure
The directory structure of new project looks like this:
```
├── configs                 <- Hydra configuration files
│   ├── trainer                 <- Configurations of Lightning trainers
│   ├── datamodule              <- Configurations of Lightning datamodules
│   ├── model                   <- Configurations of Lightning models
│   ├── callbacks               <- Configurations of Lightning callbacks
│   ├── logger                  <- Configurations of Lightning loggers
│   ├── optimizer               <- Configurations of optimizers
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
*(click to expand)*

<details>
<summary>Override any config parameter from command line</summary>

> *Hydra allows you to overwrite any parameter defined in your config, without writing any code!*
```yaml
python train.py trainer.max_epochs=20 optimizer.lr=1e-4
```

</details>


<details>
<summary>Train on CPU, GPU, TPU or even with DDP and mixed precision</summary>

> *PyTorch Lightning makes it really easy to train your models on different hardware!*
```yaml
# train on CPU
python train.py trainer.gpus=0

# train on 1 GPU
python train.py trainer.gpus=1

# train on TPU
python train.py trainer.tpu_cores=8

# train with DDP (Distributed Data Parallel) (8 GPUs, 2 nodes)
python train.py trainer.gpus=4 trainer.num_nodes=2 trainer.accelerator='ddp'

# train with mixed precision
python train.py trainer.amp_backend="apex" trainer.amp_level="O1" trainer.precision=16
```

</details>


<details>
  <summary>Train model with any logger available in PyTorch Lightning, like <a href="https://wandb.ai/">Weights&Biases</a></summary>

> *PyTorch Lightning provides convenient integrations with most popular logging frameworks. Read more [here](#experiment-tracking). Using wandb requires you to [setup account](https://www.wandb.com/) first. After that just complete the config as below.*
```yaml
# set project and entity names in `configs/logger/wandb`
wandb:
    project: "your_project_name"
    entity: "your_wandb_team_name"
```

```yaml
# train model with Weights&Biases
# link to wandb dashboard should appear in the terminal
python train.py logger=wandb
```
> *Click [here]() to see example wandb dashboard generated with this template.*

</details>


<details>
<summary>Train model with chosen experiment config</summary>

```yaml
# experiment configurations are placed in folder `configs/experiment/`
python train.py +experiment=exp_example_simple
```

</details>


<details>
<summary>Attach some callbacks to run</summary>

> *Callbacks can be used for things such as as model checkpointing, early stopping and [many more](https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html#built-in-callbacks).*
```yaml
# callback set configurations are placed in `configs/callbacks/`
python train.py callbacks=default_callbacks
```

</details>


<details>
<summary>Use different tricks available in Pytorch Lightning</summary>

> *PyTorch Lightning provides about [40+ useful trainer flags](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags).*
```yaml
# accumulate gradients from 10 training steps
python train.py trainer.accumulate_grad_batches=10

# gradient clipping may be enabled to avoid exploding gradients
python train.py trainer.gradient_clip_val=0.5

# stochastic weight averaging can make your models generalize better
python train.py trainer.stochastic_weight_avg=True

# run validation loop 4 times during a training epoch
python train.py trainer.val_check_interval=0.25
```


</details>


<details>
<summary>Easily debug</summary>

```yaml
# run 1 train, val and test loop, using only 1 batch
python train.py debug=true

# print full weight summary of all PyTorch modules
python train.py trainer.weights_summary="full"

# print execution time profiling after training ends
python train.py trainer.profiler="simple"

# try overfitting to 10 batches
python train.py trainer.overfit_batches=10

# use only 20% of the data
python train.py trainer.limit_train_batches=0.2 \
trainer.limit_val_batches=0.2 trainer.limit_test_batches=0.2
```

</details>


<details>
<summary>Resume training from checkpoint</summary>

```yaml
# checkpoint can be either path or URL
# path should be absolute!
python train.py trainer.resume_from_checkpoint="/absolute/path/to/ckpt/name.ckpt"
```
> *Currently loading ckpt in Lightning doesn't resume logger experiment, but it will be supported in future Lightning release.*

</details>


<details>
<summary>Create a sweep over hyperparameters </summary>

```yaml
# this will run 6 experiments one after the other,
# each with different combination of batch_size and learning rate
python train.py -m datamodule.batch_size=32,64,128 model.lr=0.001,0.0005
```
> *Currently sweeps aren't failure resistant (if one job crashes than the whole sweep crashes), but it will be supported in future Hydra release.*

</details>


<details>
<summary>Create a sweep over hyperparameters with Optuna</summary>

```yaml
# this will run hyperparameter search defined in `configs/config_optuna.yaml`
# over chosen experiment config
python train.py -m --config-name config_optuna.yaml +experiment=exp_example_simple
```
> *Using [Optuna Sweeper](https://hydra.cc/docs/next/plugins/optuna_sweeper) plugin doesn't require you to code any boilerplate into your pipeline, everything is defined in a single config file.*

</details>

<details>
<summary>Execute all experiments from folder</summary>

```yaml
# execute all experiments from folder `configs/experiment/`
python train.py -m '+experiment=glob(*)'
```
> *Hydra provides special syntax for controlling behavior of multiruns. Read more [here](https://hydra.cc/docs/next/tutorials/basic/running_your_app/multi-run).*

</details>

<details>
<summary>Execute sweep on a remote AWS cluster</summary>

> *This should be achievable with simple config using [Ray AWS launcher for Hydra](https://hydra.cc/docs/next/plugins/ray_launcher). Example is not yet implemented in this template.*

</details>
<br>


## Guide

### How To Start?
- First, you should probably get familiar with [PyTorch Lightning](https://www.pytorchlightning.ai)
- Next, read this blog post: [Keeping Up with PyTorch Lightning and Hydra](https://towardsdatascience.com/keeping-up-with-pytorch-lightning-and-hydra-2nd-edition-34f88e9d5c90)
- Lastly, go through [Hydra quick start guide](https://hydra.cc/docs/intro/), [basic Hydra tutorial](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/) and [docs about instantiating objects with Hydra](https://hydra.cc/docs/patterns/instantiate_objects/overview)
<br>

### Main Project Configuration
Location: [configs/config.yaml](configs/config.yaml)<br>
Main project config contains default training configuration.<br>
It determines how config is composed when simply executing command `python train.py`.<br>
It also specifies everything that shouldn't be managed by experiment configurations.
```yaml
# specify here default training configuration
defaults:
    - trainer: default_trainer.yaml
    - model: mnist_model.yaml
    - optimizer: adam.yaml
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

### Experiment Configuration
Location: [configs/experiment](configs/experiment)<br>
You should store all your experiment configurations in this folder.<br>
Experiment configurations allow you to overwrite parameters from main project configuration.

#### Simple Example
```yaml
# to execute this experiment run:
# python train.py +experiment=exp_example_simple

defaults:
    - override /trainer: default_trainer.yaml
    - override /model: mnist_model.yaml
    - override /optimizer: adam.yaml
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
    lin1_size: 128
    lin2_size: 256
    lin3_size: 64

optimizer:
    lr: 0.005

datamodule:
    batch_size: 64
    train_val_test_split: [55_000, 5_000, 10_000]
```


#### Advanced Example
```yaml
# to execute this experiment run:
# python train.py +experiment=exp_example_full

defaults:
    - override /trainer: null
    - override /model: null
    - override /optimizer: null
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
    input_size: 784
    lin1_size: 256
    dropout1: 0.30
    lin2_size: 256
    dropout2: 0.25
    lin3_size: 128
    dropout3: 0.20
    output_size: 10

optimizer:
    _target_: torch.optim.Adam
    lr: 0.001
    eps: 1e-08
    weight_decay: 0

datamodule:
    _target_: src.datamodules.mnist_datamodule.MNISTDataModule
    data_dir: ${data_dir}
    batch_size: 64
    train_val_test_split: [55_000, 5_000, 10_000]
    num_workers: 0
    pin_memory: False

logger:
    wandb:
        _target_: pytorch_lightning.loggers.wandb.WandbLogger
        project: "lightning-hydra-template"
        tags: ["best_model", "uwu"]
        notes: "Description of this model."
```
<br>

### Workflow
1. Write your PyTorch Lightning model (see [mnist_model.py](src/models/mnist_model.py) for example)
2. Write your PyTorch Lightning datamodule (see [mnist_datamodule.py](src/datamodules/mnist_datamodule.py) for example)
3. Write your experiment config, containing paths to your model and datamodule (see [configs/experiment](configs/experiment) for examples)
4. Run training with chosen experiment config:<br>
    ```bash
    python train.py +experiment=experiment_name
    ```
<br>

### Logs
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
You can change this structure by modifying paths in [main project configuration](configs/config.yaml).
<br><br>


### Experiment Tracking
PyTorch Lightning supports the most popular logging frameworks:
- Weights&Biases
- Neptune
- Comet
- MLFlow
- TestTube
- Tensorboard
- CSV

These tools help you keep track of hyperparameters and output metrics and allow you to compare and visualize results. To use one of them simply complete its configuration in [configs/logger](configs/logger) and run:
 ```yaml
 python train.py logger=logger_name
 ```
You can use many of them at once (see [configs/logger/many_loggers.yaml](configs/logger/many_loggers.yaml) for example).<br>
You can also write your own logger.<br>

Lightning provides convenient method for logging custom metrics from inside LightningModule. Read the docs [here](https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html#automatic-logging) or take a look at [MNIST example](src/models/mnist_model.py).
<br><br>


### Callbacks
Template contains example callbacks for better Weights&Biases integration (see [wandb_callbacks.py](src/callbacks/wandb_callbacks.py)).<br>

To support reproducibility:
- UploadCodeToWandbAsArtifact
- UploadCheckpointsToWandbAsArtifact
- WatchModelWithWandb

To provide examples of logging custom visualisations with callbacks only:
- LogConfusionMatrixToWandb
- LogF1PrecisionRecallHeatmapToWandb
<br>


## Best Practices

### Miniconda
Use miniconda for your python environments. Makes it easier to install cudatoolkit for GPU and PyTorch.<br>
(I find it unnecessary to install full Anaconda environment, miniconda should be enough)<br>
Example installation:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```


### Code Formating
Use pre-commit hooks to standardize code formatting of your project and save mental energy.<br>
Simply install pre-commit package with:
```yaml
pip install pre-commit
```
Next, install hooks from `.pre-commit-config.yaml`:
```
pre-commit install
```
After that your code will be automatically reformatted on every new commit.<br>
Currently `.pre-commit-config.yaml` contains configurations of **Black** (python code formatting) and **Isort** (python import sorting).
To format all files in the project use command:
```yaml
pre-commit run --all-files
```
You can exclude chosen files from automatic formatting, by modifying config (see [.pre-commit-config.yaml](pre-commit-config.yaml))


### Tests
I find myself often running into bugs that come out only in edge cases or on some specific hardware/environment. To speed up the development, I usually constantly execute simple bash scripts that run a couple of quick 1 epoch experiments, like overfitting to 10 batches, training on 25% of data, etc. You can easily modify the commands in the script for your use case. If even 1 epoch is too much for your model, then you can make it run for a couple of batches instead (by using the right trainer flags).<br>
Keep in mind those aren't real tests - it's simply executing commands one after the other, after which you need to take a look in terminal if some of them crashed.
To execute:
```yaml
bash tests/smoke_tests.sh
```


### Environment Variables
(TODO)
<br>


## Tricks
(TODO)
<!-- installing miniconda, PrettyErrors and Rich exception handling, VSCode setup,
k-fold cross validation, linter, faster tab completion import trick,
choosing metric names with '/' for wandb -->
<br>


## Other Repositories

### Inspirations
This template was inspired by:
[PyTorchLightning/deep-learninig-project-template](https://github.com/PyTorchLightning/deep-learning-project-template),
[drivendata/cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science),
[tchaton/lightning-hydra-seed](https://github.com/tchaton/lightning-hydra-seed),
[Erlemar/pytorch_tempest](https://github.com/Erlemar/pytorch_tempest),
[ryul99/pytorch-project-template](https://github.com/ryul99/pytorch-project-template),
[lucmos/nn-template](https://github.com/lucmos/nn-template).


### Useful Repositories
- [pytorch/hydra-torch](https://github.com/pytorch/hydra-torch) - resources for configuring PyTorch classes with Hydra,
- [romesco/hydra-lightning](https://github.com/romesco/hydra-lightning) - resources for configuring PyTorch Lightning classes with Hydra
- [lucmos/nn-template](https://github.com/lucmos/nn-template) - similar template that's easier to start with but less scalable

### Examples Of Repositories Using This Template
(TODO)




<br>
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
Install dependencies:
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

Train model with default configuration:
```yaml
python train.py
```

Train model with chosen logger like Weights&Biases:
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
python train.py trainer.max_epochs=20 optimizer.lr=0.0005
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
