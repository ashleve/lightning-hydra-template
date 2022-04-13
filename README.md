<div align="center">

# Lightning-Hydra-Template

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7+-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.5+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.1-89b8cd?style=for-the-badge&labelColor=gray"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>

A clean and scalable template to kickstart your deep learning project 🚀⚡🔥<br>
Click on [<kbd>Use this template</kbd>](https://github.com/ashleve/lightning-hydra-template/generate) to initialize new repository.

_Suggestions are always welcome!_

</div>

<br><br>

## 📌&nbsp;&nbsp;Introduction

This template tries to be as general as possible.

> Effective usage requires learning of a couple of technologies: [PyTorch](https://pytorch.org), [PyTorch Lightning](https://www.pytorchlightning.ai) and [Hydra](https://hydra.cc). Knowledge of some experiment logging framework like [Weights&Biases](https://wandb.com), [Neptune](https://neptune.ai) or [MLFlow](https://mlflow.org) is also recommended.

**Why you should use it:** it allows you to rapidly iterate over new models/datasets and scale your projects from small single experiments to hyperparameter searches on computing clusters, without writing any boilerplate code. To my knowledge, it's one of the most convenient all-in-one technology stack for deep learning prototyping. Quick starting point for reproducing papers, hackathons, kaggle competitions or small-team research projects. It's also a collection of best practices for efficient workflow and reproducibility.

**Why you shouldn't use it:** this template is not fitted to be a production/deployment environment, should be used more as a fast experimentation tool. Apart from that, Lightning and Hydra are still evolving and integrate many libraries, which means sometimes things break - for the list of currently known bugs, visit [this page](https://github.com/ashleve/lightning-hydra-template/labels/bug). Also, even though Lightning is very flexible, it's not well suited for every possible deep learning task. See [#Limitations](#limitations) for more.

### Why PyTorch Lightning?

[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) is a lightweight PyTorch wrapper for high-performance AI research.
It makes your code neatly organized and provides lots of useful features, like ability to run model on CPU, GPU, multi-GPU cluster and TPU.

### Why Hydra?

[Hydra](https://github.com/facebookresearch/hydra) is an open-source Python framework that simplifies the development of research and other complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line. It allows you to conveniently manage experiments and provides many useful plugins, like [Optuna Sweeper](https://hydra.cc/docs/next/plugins/optuna_sweeper) for hyperparameter search, or [Ray Launcher](https://hydra.cc/docs/next/plugins/ray_launcher) for running jobs on a cluster.

<br>

## Main Ideas Of This Template

- **Predefined Structure**: clean and scalable so that work can easily be extended and replicated | [#Project Structure](#project-structure)
- **Rapid Experimentation**: thanks to automating pipeline with config files and hydra command line superpowers | [#Your Superpowers](#your-superpowers)
- **Reproducibility**: obtaining similar results is supported in multiple ways | [#Reproducibility](#reproducibility)
- **Little Boilerplate**: so pipeline can be easily modified | [#How It Works](#how-it-works)
- **Main Configuration**: main config file specifies default training configuration | [#Main Project Configuration](#main-project-configuration)
- **Experiment Configurations**: can be composed out of smaller configs and override chosen hyperparameters | [#Experiment Configuration](#experiment-configuration)
- **Workflow**: comes down to 4 simple steps | [#Workflow](#workflow)
- **Experiment Tracking**: many logging frameworks can be easily integrated, like Tensorboard, MLFlow or W&B | [#Experiment Tracking](#experiment-tracking)
- **Logs**: all logs (checkpoints, data from loggers, hparams, etc.) are stored in a convenient folder structure imposed by Hydra | [#Logs](#logs)
- **Hyperparameter Search**: made easier with Hydra built-in plugins like [Optuna Sweeper](https://hydra.cc/docs/next/plugins/optuna_sweeper) | [#Hyperparameter Search](#hyperparameter-search)
- **Tests**: unit tests and shell/command based tests for speeding up the development | [#Tests](#tests)
- **Best Practices**: a couple of recommended tools, practices and standards for efficient workflow and reproducibility | [#Best Practices](#best-practices)

<br>

## Project Structure

The directory structure of new project looks like this:

```
├── configs                   <- Hydra configuration files
│   ├── callbacks                <- Callbacks configs
│   ├── datamodule               <- Datamodule configs
│   ├── debug                    <- Debugging configs
│   ├── experiment               <- Experiment configs
│   ├── hparams_search           <- Hyperparameter search configs
│   ├── local                    <- Local configs
│   ├── log_dir                  <- Logging directory configs
│   ├── logger                   <- Logger configs
│   ├── model                    <- Model configs
│   ├── trainer                  <- Trainer configs
│   │
│   ├── test.yaml             <- Main config for testing
│   └── train.yaml            <- Main config for training
│
├── data                   <- Project data
│
├── logs                   <- Logs generated by Hydra and PyTorch Lightning loggers
│
├── notebooks              <- Jupyter notebooks. Naming convention is a number (for ordering),
│                             the creator's initials, and a short `-` delimited description,
│                             e.g. `1.0-jqp-initial-data-exploration.ipynb`.
│
├── scripts                <- Shell scripts
│
├── src                    <- Source code
│   ├── datamodules              <- Lightning datamodules
│   ├── models                   <- Lightning models
│   ├── utils                    <- Utility scripts
│   ├── vendor                   <- Third party code that cannot be installed using PIP/Conda
│   │
│   ├── testing_pipeline.py
│   └── training_pipeline.py
│
├── tests                  <- Tests of any kind
│   ├── helpers                  <- A couple of testing utilities
│   ├── shell                    <- Shell/command based tests
│   └── unit                     <- Unit tests
│
├── test.py               <- Run testing
├── train.py              <- Run training
│
├── .env.example              <- Template of the file for storing private environment variables
├── .gitignore                <- List of files/folders ignored by git
├── .pre-commit-config.yaml   <- Configuration of pre-commit hooks for code formatting
├── requirements.txt          <- File for installing python dependencies
├── setup.cfg                 <- Configuration of linters and pytest
└── README.md
```

<br>

## 🚀&nbsp;&nbsp;Quickstart

```bash
# clone project
git clone https://github.com/ashleve/lightning-hydra-template
cd lightning-hydra-template

# [OPTIONAL] create conda environment
conda create -n myenv python=3.8
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Template contains example with MNIST classification.<br>
When running `python train.py` you should see something like this:

<div align="center">

![](https://github.com/ashleve/lightning-hydra-template/blob/resources/terminal.png)

</div>

### ⚡&nbsp;&nbsp;Your Superpowers

<details>
<summary><b>Override any config parameter from command line</b></summary>

> Hydra allows you to easily overwrite any parameter defined in your config.

```bash
python train.py trainer.max_epochs=20 model.lr=1e-4
```

> You can also add new parameters with `+` sign.

```bash
python train.py +model.new_param="uwu"
```

</details>

<details>
<summary><b>Train on CPU, GPU, multi-GPU and TPU</b></summary>

> PyTorch Lightning makes it easy to train your models on different hardware.

```bash
# train on CPU
python train.py trainer.gpus=0

# train on 1 GPU
python train.py trainer.gpus=1

# train on TPU
python train.py +trainer.tpu_cores=8

# train with DDP (Distributed Data Parallel) (4 GPUs)
python train.py trainer.gpus=4 +trainer.strategy=ddp

# train with DDP (Distributed Data Parallel) (8 GPUs, 2 nodes)
python train.py trainer.gpus=4 +trainer.num_nodes=2 +trainer.strategy=ddp
```

</details>

<details>
<summary><b>Train with mixed precision</b></summary>

```bash
# train with pytorch native automatic mixed precision (AMP)
python train.py trainer.gpus=1 +trainer.precision=16
```

</details>

<!-- deepspeed support still in beta
<details>
<summary><b>Optimize large scale models on multiple GPUs with Deepspeed</b></summary>

```bash
python train.py +trainer.
```

</details>
 -->

<details>
<summary><b>Train model with any logger available in PyTorch Lightning, like Weights&Biases or Tensorboard</b></summary>

> PyTorch Lightning provides convenient integrations with most popular logging frameworks, like Tensorboard, Neptune or simple csv files. Read more [here](#experiment-tracking). Using wandb requires you to [setup account](https://www.wandb.com/) first. After that just complete the config as below.<br> > **Click [here](https://wandb.ai/hobglob/template-dashboard/) to see example wandb dashboard generated with this template.**

```bash
# set project and entity names in `configs/logger/wandb`
wandb:
  project: "your_project_name"
  entity: "your_wandb_team_name"
```

```bash
# train model with Weights&Biases (link to wandb dashboard should appear in the terminal)
python train.py logger=wandb
```

</details>

<details>
<summary><b>Train model with chosen experiment config</b></summary>

> Experiment configurations are placed in [configs/experiment/](configs/experiment/).

```bash
python train.py experiment=example
```

</details>

<details>
<summary><b>Attach some callbacks to run</b></summary>

> Callbacks can be used for things such as as model checkpointing, early stopping and [many more](https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html#built-in-callbacks).<br>
> Callbacks configurations are placed in [configs/callbacks/](configs/callbacks/).

```bash
python train.py callbacks=default
```

</details>

<details>
<summary><b>Use different tricks available in Pytorch Lightning</b></summary>

> PyTorch Lightning provides about [40+ useful trainer flags](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags).

```yaml
# gradient clipping may be enabled to avoid exploding gradients
python train.py +trainer.gradient_clip_val=0.5

# stochastic weight averaging can make your models generalize better
python train.py +trainer.stochastic_weight_avg=true

# run validation loop 4 times during a training epoch
python train.py +trainer.val_check_interval=0.25

# accumulate gradients
python train.py +trainer.accumulate_grad_batches=10

# terminate training after 12 hours
python train.py +trainer.max_time="00:12:00:00"
```

</details>

<details>
<summary><b>Easily debug</b></summary>

> Visit [configs/debug/](configs/debug/) for different debugging configs.

```bash
# runs 1 epoch in default debugging mode
# changes logging directory to `logs/debugs/...`
# sets level of all command line loggers to 'DEBUG'
# enables extra trainer flags like tracking gradient norm
# enforces debug-friendly configuration
python train.py debug=default

# runs test epoch without training
python train.py debug=test_only

# run 1 train, val and test loop, using only 1 batch
python train.py +trainer.fast_dev_run=true

# raise exception if there are any numerical anomalies in tensors, like NaN or +/-inf
python train.py +trainer.detect_anomaly=true

# print execution time profiling after training ends
python train.py +trainer.profiler="simple"

# try overfitting to 1 batch
python train.py +trainer.overfit_batches=1 trainer.max_epochs=20

# use only 20% of the data
python train.py +trainer.limit_train_batches=0.2 \
+trainer.limit_val_batches=0.2 +trainer.limit_test_batches=0.2

# log second gradient norm of the model
python train.py +trainer.track_grad_norm=2
```

</details>

<details>
<summary><b>Resume training from checkpoint</b></summary>

> Checkpoint can be either path or URL.

```yaml
python train.py trainer.resume_from_checkpoint="/path/to/ckpt/name.ckpt"
```

> ⚠️ Currently loading ckpt in Lightning doesn't resume logger experiment, but it will be supported in future Lightning release.

</details>

<details>
<summary><b>Execute evaluation for a given checkpoint</b></summary>

> Checkpoint can be either path or URL.

```yaml
python test.py ckpt_path="/path/to/ckpt/name.ckpt"
```

</details>

<details>
<summary><b>Create a sweep over hyperparameters</b></summary>

```bash
# this will run 6 experiments one after the other,
# each with different combination of batch_size and learning rate
python train.py -m datamodule.batch_size=32,64,128 model.lr=0.001,0.0005
```

> ⚠️ This sweep is not failure resistant (if one job crashes than the whole sweep crashes).

</details>

<details>
<summary><b>Create a sweep over hyperparameters with Optuna</b></summary>

> Using [Optuna Sweeper](https://hydra.cc/docs/next/plugins/optuna_sweeper) plugin doesn't require you to code any boilerplate into your pipeline, everything is defined in a [single config file](configs/hparams_search/mnist_optuna.yaml)!

```bash
# this will run hyperparameter search defined in `configs/hparams_search/mnist_optuna.yaml`
# over chosen experiment config
python train.py -m hparams_search=mnist_optuna experiment=example_simple
```

> ⚠️ Currently this sweep is not failure resistant (if one job crashes than the whole sweep crashes). Might be supported in future Hydra release.

</details>

<details>
<summary><b>Execute all experiments from folder</b></summary>

> Hydra provides special syntax for controlling behavior of multiruns. Learn more [here](https://hydra.cc/docs/next/tutorials/basic/running_your_app/multi-run). The command below executes all experiments from folder [configs/experiment/](configs/experiment/).

```bash
python train.py -m 'experiment=glob(*)'
```

</details>

<details>
<summary><b>Execute sweep on a remote AWS cluster</b></summary>

> This should be achievable with simple config using [Ray AWS launcher for Hydra](https://hydra.cc/docs/next/plugins/ray_launcher). Example is not yet implemented in this template.

</details>

<!-- <details>
<summary><b>Execute sweep on a SLURM cluster</b></summary>

> This should be achievable with either [the right lightning trainer flags](https://pytorch-lightning.readthedocs.io/en/latest/clouds/cluster.html?highlight=SLURM#slurm-managed-cluster) or simple config using [Submitit launcher for Hydra](https://hydra.cc/docs/plugins/submitit_launcher). Example is not yet implemented in this template.

</details> -->

<details>
<summary><b>Use Hydra tab completion</b></summary>

> Hydra allows you to autocomplete config argument overrides in shell as you write them, by pressing `tab` key. Learn more [here](https://hydra.cc/docs/tutorials/basic/running_your_app/tab_completion).

</details>

<details>
<summary><b>Apply pre-commit hooks</b></summary>

> Apply pre-commit hooks to automatically format your code and configs, perform code analysis and remove output from jupyter notebooks. See [# Best Practices](#best-practices) for more.

```bash
pre-commit run -a
```

</details>

<br>

## ❤️&nbsp;&nbsp;Contributions

Have a question? Found a bug? Missing a specific feature? Have an idea for improving documentation? Feel free to file a new issue, discussion or PR with respective title and description. If you already found a solution to your problem, don't hesitate to share it. Suggestions for new best practices are always welcome!

<br>

## ℹ️&nbsp;&nbsp;Guide

### How To Get Started

- First, you should probably get familiar with [PyTorch Lightning](https://www.pytorchlightning.ai)
- Next, go through [Hydra quick start guide](https://hydra.cc/docs/intro/) and [basic Hydra tutorial](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/)

<br>

### How It Works

All PyTorch Lightning modules are dynamically instantiated from module paths specified in config. Example model config:

```yaml
_target_: src.models.mnist_model.MNISTLitModule
lr: 0.001

net:
  _target_: src.models.components.simple_dense_net.SimpleDenseNet
  input_size: 784
  lin1_size: 256
  lin2_size: 256
  lin3_size: 256
  output_size: 10
```

Using this config we can instantiate the object with the following line:

```python
model = hydra.utils.instantiate(config.model)
```

This allows you to easily iterate over new models! Every time you create a new one, just specify its module path and parameters in appropriate config file. <br>

Switch between models and datamodules with command line arguments:

```bash
python train.py model=mnist
```

The whole pipeline managing the instantiation logic is placed in [src/training_pipeline.py](src/training_pipeline.py).

<br>

### Main Project Configuration

Location: [configs/train.yaml](configs/train.yaml) <br>
Main project config contains default training configuration.<br>
It determines how config is composed when simply executing command `python train.py`.<br>

<details>
<summary><b>Show main project config</b></summary>

```yaml
# specify here default training configuration
defaults:
  - _self_
  - datamodule: mnist.yaml
  - model: mnist.yaml
  - callbacks: default.yaml
  - logger: null # set logger here or use command line (e.g. `python train.py logger=tensorboard`)
  - trainer: default.yaml
  - log_dir: default.yaml

  # experiment configs allow for version control of specific configurations
  # e.g. best hyperparameters for each combination of model and datamodule
  - experiment: null

  # debugging config (enable through command line, e.g. `python train.py debug=default)
  - debug: null

  # config for hyperparameter optimization
  - hparams_search: null

  # optional local config for machine/user specific settings
  # it's optional since it doesn't need to exist and is excluded from version control
  - optional local: default.yaml

  # enable color logging
  - override hydra/hydra_logging: colorlog
  - override hydra/job_logging: colorlog

# path to original working directory
# hydra hijacks working directory by changing it to the new log directory
# https://hydra.cc/docs/next/tutorials/basic/running_your_app/working_directory
original_work_dir: ${hydra:runtime.cwd}

# path to folder with data
data_dir: ${original_work_dir}/data/

# pretty print config at the start of the run using Rich library
print_config: True

# disable python warnings if they annoy you
ignore_warnings: True

# set False to skip model training
train: True

# evaluate on test set, using best model weights achieved during training
# lightning chooses best weights based on the metric specified in checkpoint callback
test: True

# seed for random number generators in pytorch, numpy and python.random
seed: null

# default name for the experiment, determines logging folder path
# (you can overwrite this name in experiment configs)
name: "default"
```

</details>

<br>

### Experiment Configuration

Location: [configs/experiment](configs/experiment)<br>
Experiment configs allow you to overwrite parameters from main project configuration.<br>
For example, you can use them to version control best hyperparameters for each combination of model and dataset.

<details>
<summary><b>Show example experiment config</b></summary>

```yaml
# to execute this experiment run:
# python train.py experiment=example

defaults:
  - override /datamodule: mnist.yaml
  - override /model: mnist.yaml
  - override /callbacks: default.yaml
  - override /logger: null
  - override /trainer: default.yaml

# all parameters below will be merged with parameters from default configurations set above
# this allows you to overwrite only specified parameters

# name of the run determines folder name in logs
name: "simple_dense_net"

seed: 12345

trainer:
  min_epochs: 10
  max_epochs: 10
  gradient_clip_val: 0.5

model:
  lr: 0.002
  net:
    lin1_size: 128
    lin2_size: 256
    lin3_size: 64

datamodule:
  batch_size: 64

logger:
  wandb:
    tags: ["mnist", "${name}"]
```

</details>

<br>

### Local Configuration

Location: [configs/local](configs/local) <br>
Some configurations are user/machine/installation specific (e.g. configuration of local cluster, or harddrive paths on a specific machine). For such scenarios, a file `configs/local/default.yaml` can be created which is automatically loaded but not tracked by Git.

<details>
<summary><b>Show example local Slurm cluster config</b></summary>

```yaml
# @package _global_

defaults:
  - override /hydra/launcher@_here_: submitit_slurm

data_dir: /mnt/scratch/data/

hydra:
  launcher:
    timeout_min: 1440
    gpus_per_task: 1
    gres: gpu:1
  job:
    env_set:
      MY_VAR: /home/user/my/system/path
      MY_KEY: asdgjhawi8y23ihsghsueity23ihwd
```

</details>

<br>

### Workflow

1. Write your PyTorch Lightning module (see [models/mnist_module.py](src/models/mnist_module.py) for example)
2. Write your PyTorch Lightning datamodule (see [datamodules/mnist_datamodule.py](src/datamodules/mnist_datamodule.py) for example)
3. Write your experiment config, containing paths to your model and datamodule
4. Run training with chosen experiment config: `python train.py experiment=experiment_name`

<br>

### Logs

**Hydra creates new working directory for every executed run.** By default, logs have the following structure:

```
├── logs
│   ├── experiments                     # Folder for the logs generated by experiments
│   │   ├── runs                          # Folder for single runs
│   │   │   ├── experiment_name             # Experiment name
│   │   │   │   ├── YYYY-MM-DD_HH-MM-SS       # Datetime of the run
│   │   │   │   │   ├── .hydra                  # Hydra logs
│   │   │   │   │   ├── csv                     # Csv logs
│   │   │   │   │   ├── wandb                   # Weights&Biases logs
│   │   │   │   │   ├── checkpoints             # Training checkpoints
│   │   │   │   │   └── ...                     # Any other thing saved during training
│   │   │   │   └── ...
│   │   │   └── ...
│   │   │
│   │   └── multiruns                     # Folder for multiruns
│   │       ├── experiment_name             # Experiment name
│   │       │   ├── YYYY-MM-DD_HH-MM-SS       # Datetime of the multirun
│   │       │   │   ├──1                        # Multirun job number
│   │       │   │   ├──2
│   │       │   │   └── ...
│   │       │   └── ...
│   │       └── ...
│   │
│   ├── evaluations                       # Folder for the logs generated during testing
│   │   └── ...
│   │
│   └── debugs                            # Folder for the logs generated during debugging
│       └── ...
```

You can change this structure by modifying paths in [hydra configuration](configs/log_dir).

<br>

### Experiment Tracking

PyTorch Lightning supports many popular logging frameworks:<br>
**[Weights&Biases](https://www.wandb.com/) · [Neptune](https://neptune.ai/) · [Comet](https://www.comet.ml/) · [MLFlow](https://mlflow.org) · [Tensorboard](https://www.tensorflow.org/tensorboard/)**

These tools help you keep track of hyperparameters and output metrics and allow you to compare and visualize results. To use one of them simply complete its configuration in [configs/logger](configs/logger) and run:

```bash
python train.py logger=logger_name
```

You can use many of them at once (see [configs/logger/many_loggers.yaml](configs/logger/many_loggers.yaml) for example).

You can also write your own logger.

Lightning provides convenient method for logging custom metrics from inside LightningModule. Read the docs [here](https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html#automatic-logging) or take a look at [MNIST example](src/models/mnist_module.py).

<br>

### Hyperparameter Search

Defining hyperparameter optimization is as easy as adding new config file to [configs/hparams_search](configs/hparams_search).

<details>
<summary><b>Show example</b></summary>

```yaml
defaults:
  - override /hydra/sweeper: optuna

# choose metric which will be optimized by Optuna
optimized_metric: "val/acc_best"

hydra:
  # here we define Optuna hyperparameter search
  # it optimizes for value returned from function with @hydra.main decorator
  # learn more here: https://hydra.cc/docs/next/plugins/optuna_sweeper
  sweeper:
    _target_: hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper
    storage: null
    study_name: null
    n_jobs: 1

    # 'minimize' or 'maximize' the objective
    direction: maximize

    # number of experiments that will be executed
    n_trials: 20

    # choose Optuna hyperparameter sampler
    # learn more here: https://optuna.readthedocs.io/en/stable/reference/samplers.html
    sampler:
      _target_: optuna.samplers.TPESampler
      seed: 12345
      consider_prior: true
      prior_weight: 1.0
      consider_magic_clip: true
      consider_endpoints: false
      n_startup_trials: 10
      n_ei_candidates: 24
      multivariate: false
      warn_independent_sampling: true

    # define range of hyperparameters
    search_space:
      datamodule.batch_size:
        type: categorical
        choices: [32, 64, 128]
      model.lr:
        type: float
        low: 0.0001
        high: 0.2
      model.net.lin1_size:
        type: categorical
        choices: [32, 64, 128, 256, 512]
      model.net.lin2_size:
        type: categorical
        choices: [32, 64, 128, 256, 512]
      model.net.lin3_size:
        type: categorical
        choices: [32, 64, 128, 256, 512]
```

</details>

Next, you can execute it with: `python train.py -m hparams_search=mnist_optuna`

Using this approach doesn't require you to add any boilerplate into your pipeline, everything is defined in a single config file.

You can use different optimization frameworks integrated with Hydra, like Optuna, Ax or Nevergrad.

The `optimization_results.yaml` will be available under `logs/multirun` folder.

This approach doesn't support advanced technics like prunning - for more sophisticated search, you probably shouldn't use hydra multirun feature and instead write your own optimization pipeline.

<br>

### Inference

The following code is an example of loading model from checkpoint and running predictions.<br>

<details>
<summary><b>Show example</b></summary>

```python
from PIL import Image
from torchvision import transforms

from src.models.mnist_module import MNISTLitModule


def predict():
    """Example of inference with trained model.
    It loads trained image classification model from checkpoint.
    Then it loads example image and predicts its label.
    """

    # ckpt can be also a URL!
    CKPT_PATH = "last.ckpt"

    # load model from checkpoint
    # model __init__ parameters will be loaded from ckpt automatically
    # you can also pass some parameter explicitly to override it
    trained_model = MNISTLitModule.load_from_checkpoint(checkpoint_path=CKPT_PATH)

    # print model hyperparameters
    print(trained_model.hparams)

    # switch to evaluation mode
    trained_model.eval()
    trained_model.freeze()

    # load data
    img = Image.open("data/example_img.png").convert("L")  # convert to black and white
    # img = Image.open("data/example_img.png").convert("RGB")  # convert to RGB

    # preprocess
    mnist_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize((28, 28)),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    img = mnist_transforms(img)
    img = img.reshape((1, *img.size()))  # reshape to form batch of size 1

    # inference
    output = trained_model(img)
    print(output)


if __name__ == "__main__":
    predict()

```

</details>

<br>

### Tests

Template comes with example tests implemented with pytest library. To execute them simply run:

```bash
# run all tests
pytest

# run tests from specific file
pytest tests/shell/test_basic_commands.py

# run all tests except the ones marked as slow
pytest -k "not slow"
```

To speed up the development, you can once in a while execute tests that run a couple of quick experiments, like training 1 epoch on 25% of data, executing single train/val/test step, etc. Those kind of tests don't check for any specific output - they exist to simply verify that executing some bash commands doesn't end up in throwing exceptions. You can find them implemented in [tests/shell](tests/shell) folder.

You can easily modify the commands in the scripts for your use case. If 1 epoch is too much for your model, then make it run for a couple of batches instead (by using the right trainer flags).

<br>

### Callbacks

The branch [`wandb-callbacks`](https://github.com/ashleve/lightning-hydra-template/tree/wandb-callbacks) contains example callbacks enabling better Weights&Biases integration, which you can use as a reference for writing your own callbacks (see [wandb_callbacks.py](https://github.com/ashleve/lightning-hydra-template/tree/wandb-callbacks/src/callbacks/wandb_callbacks.py)).

Callbacks which support reproducibility:

- **WatchModel**
- **UploadCodeAsArtifact**
- **UploadCheckpointsAsArtifact**

Callbacks which provide examples of logging custom visualisations:

- **LogConfusionMatrix**
- **LogF1PrecRecHeatmap**
- **LogImagePredictions**

To try all of the callbacks at once, switch to the right branch:

```bash
git checkout wandb-callbacks
```

And then run the following command:

```bash
python train.py logger=wandb callbacks=wandb
```

To see the result of all the callbacks attached, take a look at [this experiment dashboard](https://wandb.ai/hobglob/template-tests/runs/3rw7q70h).

<br>

### Multi-GPU Training

Lightning supports multiple ways of doing distributed training. The most common one is DDP, which spawns separate process for each GPU and averages gradients between them. To learn about other approaches read the [lightning docs](https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html).

You can run DDP on mnist example with 4 GPUs like this:

```bash
python train.py trainer.gpus=4 +trainer.strategy=ddp
```

⚠️ When using DDP you have to be careful how you write your models - learn more [here](https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html).

<br>

### Docker

First you will need to [install Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to enable GPU support.

The template Dockerfile is provided on branch [`dockerfiles`](https://github.com/ashleve/lightning-hydra-template/tree/dockerfiles). Copy it to the template root folder.

To build the container use:

```bash
docker build -t <project_name> .
```

To mount the project to the container use:

```bash
docker run -v $(pwd):/workspace/project --gpus all -it --rm <project_name>
```

<br>

### Reproducibility

What provides reproducibility:

- Hydra manages your configs.
- Hydra manages your logging paths and makes every executed run store its hyperparameters and config overrides in a separate file in logs.
- LightningDataModule allows you to encapsulate data split, transformations and default parameters in a single, clean abstraction.
- LightningModule separates your research code from engineering code in a clean way.
- Experiment tracking frameworks take care of logging metrics and hparams, some can also store results and artifacts in cloud.
- Pytorch Lightning takes care of creating training checkpoints.
- [Example callbacks for wandb](https://github.com/ashleve/lightning-hydra-template/tree/wandb-callbacks) show how you can save and upload a snapshot of codebase every time the run is executed, as well as upload ckpts and track model gradients.

<!--
You can load the config of previous run using:

```bash
python train.py --config-path /logs/runs/.../.hydra/ --config-name config.yaml
```

The `config.yaml` from `.hydra` folder contains all overriden parameters and sections. This approach however is not officially supported by Hydra and doesn't override the `hydra/` part of the config, meaning logging paths will revert to default!
 -->
<br>

### Limitations

- Currently, template doesn't support k-fold cross validation, but it's possible to achieve it with Lightning Loop interface. See the [official example](https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/loop_examples/kfold.py). Implementing it requires rewriting the training pipeline.
- Currently hyperparameter search with Hydra Optuna Plugin doesn't support prunning.
- Hydra changes working directory to new logging folder for every executed run, which might not be compatible with the way some libraries work.
- Restoring logger state is currently not supported. This might change in future lightning realease.

<br>

## Useful Tricks

<details>
<summary><b>Accessing datamodule attributes in model</b></summary>

1. The simplest way is to pass datamodule attribute directly to model on initialization:

   ```python
   # ./src/training_pipeline.py
   datamodule = hydra.utils.instantiate(config.datamodule)
   model = hydra.utils.instantiate(config.model, some_param=datamodule.some_param)
   ```

   This is not a very robust solution, since it assumes all your datamodules have `some_param` attribute available (otherwise the run will crash).

2. If you only want to access datamodule config, you can simply pass it as an init parameter:

   ```python
   # ./src/training_pipeline.py
   model = hydra.utils.instantiate(config.model, dm_conf=config.datamodule, _recursive_=False)
   ```

   Now you can access any datamodule config part like this:

   ```python
   # ./src/models/my_model.py
   class MyLitModel(LightningModule):
   	def __init__(self, dm_conf, param1, param2):
   		super().__init__()

   		batch_size = dm_conf.batch_size
   ```

3. If you need to access the datamodule object attributes, a little hacky solution is to add Omegaconf resolver to your datamodule:

   ```python
   # ./src/datamodules/my_datamodule.py
   from omegaconf import OmegaConf

   class MyDataModule(LightningDataModule):
   	def __init__(self, param1, param2):
   		super().__init__()

   		self.param1 = param1

   		resolver_name = "datamodule"
   		OmegaConf.register_new_resolver(
   			resolver_name,
   			lambda name: getattr(self, name),
   			use_cache=False
   		)
   ```

   This way you can reference any datamodule attribute from your config like this:

   ```yaml
   # this will return attribute 'param1' from datamodule object
   param1: ${datamodule:param1}
   ```

   When later accessing this field, say in your lightning model, it will get automatically resolved based on all resolvers that are registered. Remember not to access this field before datamodule is initialized or it will crash.

   **You also need to set `resolve=False` in `print_config(...)` in [utils](src/urils/__init__.py) to prevent config printing from accessing the parameter before datamodule is initialized:**

   ```python
   # ./src/urils/__init__.py
    def extras(config: DictConfig) -> None:

      ...

      log.info("Printing config tree with Rich! <config.print_config=True>")
      print_config(config, resolve=False)

      ...
   ```

</details>

<details>
<summary><b>Automatic activation of virtual environment and tab completion when entering folder</b></summary>

1. Create a new file called `.autoenv` (this name is excluded from version control in `.gitignore`). <br>
   You can use it to automatically execute shell commands when entering folder. Add some commands to your `.autoenv` file, like in the example below:

   ```bash
   # activate conda environment
   conda activate myenv

   # activate hydra tab completion for bash
   eval "$(python train.py -sc install=bash)"
   ```

   (these commands will be executed whenever you're openning or switching terminal to folder containing `.autoenv` file)

2. To setup this automation for bash, execute the following line (it will append your `.bashrc` file):

   ```bash
   echo "autoenv() { if [ -x .autoenv ]; then source .autoenv ; echo '.autoenv executed' ; fi } ; cd() { builtin cd \"\$@\" ; autoenv ; } ; autoenv" >> ~/.bashrc
   ```

3. Lastly add execution previliges to your `.autoenv` file:

   ```
   chmod +x .autoenv
   ```

   (for safety, only `.autoenv` with previligies will be executed)

**Explanation**

The mentioned line appends your `.bashrc` file with 2 commands:

1. `autoenv() { if [ -x .autoenv ]; then source .autoenv ; echo '.autoenv executed' ; fi }` - this declares the `autoenv()` function, which executes `.autoenv` file if it exists in current work dir and has execution previligies
2. `cd() { builtin cd \"\$@\" ; autoenv ; } ; autoenv` - this extends behaviour of `cd` command, to make it execute `autoenv()` function each time you change folder in terminal or open new terminal

</details>

<!--
<details>
<summary><b>Making sweeps failure resistant</b></summary>

TODO

</details>
 -->

<br>

## Best Practices

<details>
<summary><b>Use Miniconda for GPU environments</b></summary>

Use miniconda for your python environments (it's usually unnecessary to install full anaconda environment, miniconda should be enough).
It makes it easier to install some dependencies, like cudatoolkit for GPU support. It also allows you to acccess your environments globally.

Example installation:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Create new conda environment:

```bash
conda create -n myenv python=3.8
conda activate myenv
```

</details>

<details>
<summary><b>Use automatic code formatting</b></summary>

Use pre-commit hooks to standardize code formatting of your project and save mental energy.<br>
Simply install pre-commit package with:

```bash
pip install pre-commit
```

Next, install hooks from [.pre-commit-config.yaml](.pre-commit-config.yaml):

```bash
pre-commit install
```

After that your code will be automatically reformatted on every new commit.<br>
Currently template contains configurations of **black** (python code formatting), **isort** (python import sorting), **docformatter** (python docstring formatting), **flake8** (python code analysis), **prettier** (yaml formating) and **nbstripout** (clearing output from jupyter notebooks). <br>

To reformat all files in the project use command:

```bash
pre-commit run -a
```

</details>

<details>
<summary><b>Set private environment variables in .env file</b></summary>

System specific variables (e.g. absolute paths to datasets) should not be under version control or it will result in conflict between different users. Your private keys also shouldn't be versioned since you don't want them to be leaked.<br>

Template contains `.env.example` file, which serves as an example. Create a new file called `.env` (this name is excluded from version control in .gitignore).
You should use it for storing environment variables like this:

```
MY_VAR=/home/user/my_system_path
```

All variables from `.env` are loaded in `train.py` automatically.

Hydra allows you to reference any env variable in `.yaml` configs like this:

```yaml
path_to_data: ${oc.env:MY_VAR}
```

</details>

<details>
<summary><b>Name metrics using '/' character</b></summary>

Depending on which logger you're using, it's often useful to define metric name with `/` character:

```python
self.log("train/loss", loss)
```

This way loggers will treat your metrics as belonging to different sections, which helps to get them organised in UI.

</details>

<details>
<summary><b>Use torchmetrics</b></summary>

Use official [torchmetrics](https://github.com/PytorchLightning/metrics) library to ensure proper calculation of metrics. This is especially important for multi-GPU training!

For example, instead of calculating accuracy by yourself, you should use the provided `Accuracy` class like this:

```python
from torchmetrics.classification.accuracy import Accuracy


class LitModel(LightningModule):
    def __init__(self)
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def training_step(self, batch, batch_idx):
        ...
        acc = self.train_acc(predictions, targets)
        self.log("train/acc", acc)
        ...

    def validation_step(self, batch, batch_idx):
        ...
        acc = self.val_acc(predictions, targets)
        self.log("val/acc", acc)
        ...
```

Make sure to use different metric instance for each step to ensure proper value reduction over all GPU processes.

Torchmetrics provides metrics for most use cases, like F1 score or confusion matrix. Read [documentation](https://torchmetrics.readthedocs.io/en/latest/#more-reading) for more.

</details>

<details>
<summary><b>Follow PyTorch Lightning style guide</b></summary>

The style guide is available [here](https://pytorch-lightning.readthedocs.io/en/latest/starter/style_guide.html).<br>

1. Be explicit in your init. Try to define all the relevant defaults so that the user doesn’t have to guess. Provide type hints. This way your module is reusable across projects!

   ```python
   class LitModel(LightningModule):
       def __init__(self, layer_size: int = 256, lr: float = 0.001):
   ```

2. Preserve the recommended method order.

   ```python
   class LitModel(LightningModule):

       def __init__():
           ...

       def forward():
           ...

       def training_step():
           ...

       def training_step_end():
           ...

       def training_epoch_end():
           ...

       def validation_step():
           ...

       def validation_step_end():
           ...

       def validation_epoch_end():
           ...

       def test_step():
           ...

       def test_step_end():
           ...

       def test_epoch_end():
           ...

       def configure_optimizers():
           ...

       def any_extra_hook():
           ...
   ```

</details>

<details>
<summary><b>Version control your data and models with DVC</b></summary>

Use [DVC](https://dvc.org) to version control big files, like your data or trained ML models.<br>
To initialize the dvc repository:

```bash
dvc init
```

To start tracking a file or directory, use `dvc add`:

```bash
dvc add data/MNIST
```

DVC stores information about the added file (or a directory) in a special .dvc file named data/MNIST.dvc, a small text file with a human-readable format. This file can be easily versioned like source code with Git, as a placeholder for the original data:

```bash
git add data/MNIST.dvc data/.gitignore
git commit -m "Add raw data"
```

</details>

<details>
<summary><b>Support installing project as a package</b></summary>

It allows other people to easily use your modules in their own projects.
Change name of the `src` folder to your project name and add `setup.py` file:

```python
from setuptools import find_packages, setup


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
        "hydra-core>=1.1.0",
    ],
    packages=find_packages(),
)
```

Now your project can be installed from local files:

```bash
pip install -e .
```

Or directly from git repository:

```bash
pip install git+git://github.com/YourGithubName/your-repo-name.git --upgrade
```

So any file can be easily imported into any other file like so:

```python
from project_name.models.mnist_module import MNISTLitModule
from project_name.datamodules.mnist_datamodule import MNISTDataModule
```

</details>

<!-- <details>
<summary><b>Make notebooks independent from other files</b></summary>

It's a good practice for jupyter notebooks to be portable. Try to make them independent from src files. If you need to access external code, try to embed it inside the notebook.

</details> -->

<!--<details>
<summary><b>Use Docker</b></summary>

Docker makes it easy to initialize the whole training environment, e.g. when you want to execute experiments in cloud or on some private computing cluster. You can extend [dockerfiles](https://github.com/ashleve/lightning-hydra-template/tree/dockerfiles) provided in the template with your own instructions for building the image.<br>

</details> -->

<br>

## Other Repositories

<details>
<summary><b>Inspirations</b></summary>

This template was inspired by:
[PyTorchLightning/deep-learninig-project-template](https://github.com/PyTorchLightning/deep-learning-project-template),
[drivendata/cookiecutter-data-science](https://github.com/drivendata/cookiecutter-data-science),
[tchaton/lightning-hydra-seed](https://github.com/tchaton/lightning-hydra-seed),
[Erlemar/pytorch_tempest](https://github.com/Erlemar/pytorch_tempest),
[lucmos/nn-template](https://github.com/lucmos/nn-template).

</details>

<details>
<summary><b>Useful repositories</b></summary>

- [pytorch/hydra-torch](https://github.com/pytorch/hydra-torch) - resources for configuring PyTorch classes with Hydra,
- [romesco/hydra-lightning](https://github.com/romesco/hydra-lightning) - resources for configuring PyTorch Lightning classes with Hydra
- [lucmos/nn-template](https://github.com/lucmos/nn-template) - similar template
- [PyTorchLightning/lightning-transformers](https://github.com/PyTorchLightning/lightning-transformers) - official Lightning Transformers repo built with Hydra

</details>

<!-- ## :star:&nbsp; Stargazers Over Time
[![Stargazers over time](https://starchart.cc/ashleve/lightning-hydra-template.svg)](https://starchart.cc/ashleve/lightning-hydra-template) -->

<br>

## License

This project is licensed under the MIT License.

```
MIT License

Copyright (c) 2021 ashleve

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

<br>
<br>
<br>
<br>

**DELETE EVERYTHING ABOVE FOR YOUR PROJECT**

---

<div align="center">

# Your Project Name

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

What it does

## How to run

Install dependencies

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.8
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

Train model with default configuration

```bash
# train on CPU
python train.py trainer.gpus=0

# train on GPU
python train.py trainer.gpus=1
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python train.py trainer.max_epochs=20 datamodule.batch_size=64
```
