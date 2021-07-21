<div align="center">

# Lightning-Hydra-Template


<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.7--3.9-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/-PyTorch 1.8+-ee4c2c?style=for-the-badge&logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning 1.3+-792ee5?style=for-the-badge&logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: hydra" src="https://img.shields.io/badge/config-hydra 1.1-89b8cd?style=for-the-badge&labelColor=gray"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=for-the-badge&labelColor=gray"></a>
<!-- <a href="https://hub.docker.com/r/ashlev/lightning-hydra"><img alt="Docker" src="https://img.shields.io/badge/docker-257bd6?style=for-the-badge&logo=docker&logoColor=white"></a> -->

A clean and scalable template to kickstart your deep learning project 🚀⚡🔥<br>
Click on [<kbd>Use this template</kbd>](https://github.com/ashleve/lightning-hydra-template/generate) to initialize new repository.

*Suggestions are always welcome!*

</div>
<br><br>

<!--
If you use this template please add <br>
[![](https://shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=303030)](https://github.com/ashleve/lightning-hydra-template) <br>
to your `README.md`.
<br><br>
-->

## 📌&nbsp;&nbsp;Introduction
This template tries to be as general as possible - you can easily delete any unwanted features from the pipeline or rewire the configuration, by modifying behavior in [src/train.py](src/train.py).

> Effective usage of this template requires learning of a couple of technologies: [PyTorch](https://pytorch.org), [PyTorch Lightning](https://www.pytorchlightning.ai) and [Hydra](https://hydra.cc). Knowledge of some experiment logging framework like [Weights&Biases](https://wandb.com), [Neptune](https://neptune.ai) or [MLFlow](https://mlflow.org) is also recommended.

**Why you should use it:** it allows you to rapidly iterate over new models/datasets and scale your projects from small single experiments to hyperparameter searches on computing clusters, without writing any boilerplate code. To my knowledge, it's one of the most convenient all-in-one technology stack for Deep Learning research. Good starting point for reproducing papers, kaggle competitions or small-team research projects. It's also a collection of best practices for efficient workflow and reproducibility.

**Why you shouldn't use it:** Lightning and Hydra are not yet mature, which means you might run into some bugs sooner or later. Also, even though Lightning is very flexible, it's not well suited for every possible deep learning task.

### Why PyTorch Lightning?
[PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) is a lightweight PyTorch wrapper for high-performance AI research.
It makes your code neatly organized and provides lots of useful features, like ability to run model on CPU, GPU, multi-GPU cluster and TPU.


### Why Hydra?
[Hydra](https://github.com/facebookresearch/hydra) is an open-source Python framework that simplifies the development of research and other complex applications. The key feature is the ability to dynamically create a hierarchical configuration by composition and override it through config files and the command line. It  allows you to conveniently manage experiments and provides many useful plugins, like [Optuna Sweeper](https://hydra.cc/docs/next/plugins/optuna_sweeper) for hyperparameter search, or [Ray Launcher](https://hydra.cc/docs/next/plugins/ray_launcher) for running jobs on a cluster.
<br>
<br>
<br>


## Main Ideas Of This Template
- **Predefined Structure**: clean and scalable so that work can easily be extended and replicated (see [#Project Structure](#project-structure))
- **Rapid Experimentation**: thanks to automating pipeline with config files and hydra command line superpowers (see [#Your Superpowers](#your-superpowers))
- **Little Boilerplate**: so pipeline can be easily modified (see [src/train.py](src/train.py))
- **Main Configuration**: main config file specifies default training configuration (see [#Main Project Configuration](#main-project-configuration))
- **Experiment Configurations**: stored in a separate folder, they can be composed out of smaller configs, override chosen parameters or define everything from scratch (see [#Experiment Configuration](#experiment-configuration))
- **Workflow**: comes down to 4 simple steps (see [#Workflow](#workflow))
- **Experiment Tracking**: many logging frameworks can be easily integrated! (see [#Experiment Tracking](#experiment-tracking))
- **Logs**: all logs (checkpoints, data from loggers, chosen hparams, etc.) are stored in a convenient folder structure imposed by Hydra (see [#Logs](#logs))
- **Hyperparameter Search**: made easier with Hydra built in plugins like [Optuna Sweeper](https://hydra.cc/docs/next/plugins/optuna_sweeper) (see [#Hyperparameter Search](#hyperparameter-search))
- **Tests**: unit tests and command based tests (see [#Tests](#tests))
- **Extra Features**: optional utilities to make your life easier (see [#Extra Features](#extra-features))
- **Best Practices**: a couple of recommended tools, practices and standards for efficient workflow and reproducibility (see [#Best Practices](#best-practices))
<br>


## Project Structure
The directory structure of new project looks like this:
```
├── bash                    <- Bash scripts
│   ├── setup_conda.sh          <- Setup conda environment
│   └── schedule.sh             <- Schedule execution of many runs
│
├── configs                 <- Hydra configuration files
│   ├── callbacks               <- Callbacks configs
│   ├── datamodule              <- Datamodule configs
│   ├── experiment              <- Experiment configs
│   ├── hparams_search          <- Hyperparameter search configs
│   ├── hydra                   <- Hydra related configs
│   ├── logger                  <- Logger configs
│   ├── model                   <- Model configs
│   ├── trainer                 <- Trainer configs
│   │
│   └── config.yaml             <- Main project configuration file
│
├── data                    <- Project data
│
├── logs                    <- Logs generated by Hydra and PyTorch Lightning loggers
│
├── notebooks               <- Jupyter notebooks. Naming convention is a number (for ordering),
│                              the creator's initials, and a short `-` delimited description, e.g.
│                              `1.0-jqp-initial-data-exploration.ipynb`.
│
├── tests                   <- Tests of any kind
│   ├── helpers                 <- A couple of testing utilities
│   ├── shell                   <- Shell/command based tests
│   └── unit                    <- Unit tests
│
├── src
│   ├── callbacks               <- Lightning callbacks
│   ├── datamodules             <- Lightning datamodules
│   ├── models                  <- Lightning models
│   ├── utils                   <- Utility scripts
│   │
│   └── train.py                <- Training pipeline
│
├── run.py                  <- Run pipeline with chosen experiment configuration
│
├── .env.example            <- Template of the file for storing private environment variables
├── .gitignore              <- List of files/folders ignored by git
├── .pre-commit-config.yaml <- Configuration of automatic code formatting
├── setup.cfg               <- Configurations of linters and pytest
├── Dockerfile              <- File for building docker container
├── requirements.txt        <- File for installing python dependencies
├── LICENSE
└── README.md
```
<br>


## 🚀&nbsp;&nbsp;Quickstart
```yaml
# clone project
git clone https://github.com/ashleve/lightning-hydra-template
cd lightning-hydra-template

# [OPTIONAL] create conda environment
bash bash/setup_conda.sh

# install requirements
pip install -r requirements.txt
```

Template contains example with MNIST classification.<br>
When running `python run.py` you should see something like this:
<div align="center">

![](https://github.com/ashleve/lightning-hydra-template/blob/resources/terminal.png)

</div>

### ⚡&nbsp;&nbsp;Your Superpowers
(click to expand)

<details>
<summary><b>Override any config parameter from command line</b></summary>

> Hydra allows you to easily overwrite any parameter defined in your config.
```yaml
python run.py trainer.max_epochs=20 model.lr=1e-4
```
> You can also add new parameters with `+` sign.
```yaml
python run.py +model.new_param="uwu"

```

</details>


<details>
<summary><b>Train on CPU, GPU, multi-GPU and TPU</b></summary>

> PyTorch Lightning makes it easy to train your models on different hardware.
```yaml
# train on CPU
python run.py trainer.gpus=0

# train on 1 GPU
python run.py trainer.gpus=1

# train on TPU
python run.py +trainer.tpu_cores=8

# train with DDP (Distributed Data Parallel) (4 GPUs)
python run.py trainer.gpus=4 +trainer.accelerator='ddp'

# train with DDP (Distributed Data Parallel) (8 GPUs, 2 nodes)
python run.py trainer.gpus=4 +trainer.num_nodes=2 +trainer.accelerator='ddp'
```

</details>


<details>
<summary><b>Train with mixed precision</b></summary>

```yaml
# train with mixed precision (Apex level O1)
python run.py trainer.gpus=1 +trainer.amp_backend="apex" +trainer.precision=16 \
+trainer.amp_level="O1"

# train with mixed precision (Apex level O2)
python run.py trainer.gpus=1 +trainer.amp_backend="apex" +trainer.precision=16 \
+trainer.amp_level="O2"
```

</details>


<details>
  <summary><b>Train model with any logger available in PyTorch Lightning, like Weights&Biases</b></summary>

> PyTorch Lightning provides convenient integrations with most popular logging frameworks. Read more [here](#experiment-tracking). Using wandb requires you to [setup account](https://www.wandb.com/) first. After that just complete the config as below.<br>
**Click [here](https://wandb.ai/hobglob/template-dashboard/) to see example wandb dashboard generated with this template.**
```yaml
# set project and entity names in `configs/logger/wandb`
wandb:
    project: "your_project_name"
    entity: "your_wandb_team_name"
```

```yaml
# train model with Weights&Biases
# link to wandb dashboard should appear in the terminal
python run.py logger=wandb
```

</details>


<details>
<summary><b>Train model with chosen experiment config</b></summary>

> Experiment configurations are placed in [configs/experiment/](configs/experiment/).
```yaml
python run.py experiment=example_simple
```

</details>


<details>
<summary><b>Attach some callbacks to run</b></summary>

> Callbacks can be used for things such as as model checkpointing, early stopping and [many more](https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html#built-in-callbacks).<br>
Callbacks configurations are placed in [configs/callbacks/](configs/callbacks/).
```yaml
python run.py callbacks=default
```

</details>


<details>
<summary><b>Use different tricks available in Pytorch Lightning</b></summary>

> PyTorch Lightning provides about [40+ useful trainer flags](https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags).
```yaml
# gradient clipping may be enabled to avoid exploding gradients
python run.py +trainer.gradient_clip_val=0.5

# stochastic weight averaging can make your models generalize better
python run.py +trainer.stochastic_weight_avg=true

# run validation loop 4 times during a training epoch
python run.py +trainer.val_check_interval=0.25

# accumulate gradients
python run.py +trainer.accumulate_grad_batches=10

# terminate training after 12 hours
python run.py +trainer.max_time="00:12:00:00"
```

</details>


<details>
<summary><b>Easily debug</b></summary>

```yaml
# run 1 train, val and test loop, using only 1 batch
python run.py debug=true

# print full weight summary of all PyTorch modules
python run.py trainer.weights_summary="full"

# print execution time profiling after training ends
python run.py +trainer.profiler="simple"

# raise exception, if any of the parameters or the loss are NaN or +/-inf
python run.py +trainer.terminate_on_nan=true

# try overfitting to 1 batch
python run.py +trainer.overfit_batches=1 trainer.max_epochs=20

# use only 20% of the data
python run.py +trainer.limit_train_batches=0.2 \
+trainer.limit_val_batches=0.2 +trainer.limit_test_batches=0.2

# log second gradient norm of the model
python run.py +trainer.track_grad_norm=2
```

</details>


<details>
<summary><b>Resume training from checkpoint</b></summary>

> Checkpoint can be either path or URL. Path should be absolute!
```yaml
python run.py +trainer.resume_from_checkpoint="/absolute/path/to/ckpt/name.ckpt"
```
> ⚠️ Currently loading ckpt in Lightning doesn't resume logger experiment, but it will be supported in future Lightning release.

</details>


<details>
<summary><b>Create a sweep over hyperparameters</b></summary>

```yaml
# this will run 6 experiments one after the other,
# each with different combination of batch_size and learning rate
python run.py -m datamodule.batch_size=32,64,128 model.lr=0.001,0.0005
```
> ⚠️ Currently sweeps aren't failure resistant (if one job crashes than the whole sweep crashes), but it will be supported in future Hydra release.

</details>


<details>
<summary><b>Create a sweep over hyperparameters with Optuna</b></summary>

> Using [Optuna Sweeper](https://hydra.cc/docs/next/plugins/optuna_sweeper) plugin doesn't require you to code any boilerplate into your pipeline, everything is defined in a [single config file](configs/hparams_search/mnist_optuna.yaml)!
```yaml
# this will run hyperparameter search defined in `configs/hparams_search/mnist_optuna.yaml`
# over chosen experiment config
python run.py -m hparams_search=mnist_optuna experiment=example_simple
```

</details>

<details>
<summary><b>Execute all experiments from folder</b></summary>

> Hydra provides special syntax for controlling behavior of multiruns. Learn more [here](https://hydra.cc/docs/next/tutorials/basic/running_your_app/multi-run). The command below executes all experiments from folder [configs/experiment/](configs/experiment/).
```yaml
python run.py -m 'experiment=glob(*)'
```

</details>

<details>
<summary><b>Execute sweep on a remote AWS cluster</b></summary>

> This should be achievable with simple config using [Ray AWS launcher for Hydra](https://hydra.cc/docs/next/plugins/ray_launcher). Example is not yet implemented in this template.

</details>

<details>
<summary><b>Execute sweep on a SLURM cluster</b></summary>

> This should be achievable with either [the right lightning trainer flags](https://pytorch-lightning.readthedocs.io/en/latest/clouds/cluster.html?highlight=SLURM#slurm-managed-cluster) or simple config using [Submitit launcher for Hydra](https://hydra.cc/docs/plugins/submitit_launcher). Example is not yet implemented in this template.

</details>


<details>
<summary><b>Use Hydra tab completion</b></summary>

> Hydra allows you to autocomplete config argument overrides in shell as you write them, by pressing `tab` key. Learn more [here](https://hydra.cc/docs/tutorials/basic/running_your_app/tab_completion).

</details>
<br>


## 🐳&nbsp;&nbsp;Docker
First you will need to [install Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) to enable GPU  support. <br>
To build the container from provided Dockerfile use:
```bash
docker build -t <project_name> .
```
To mount the project to the container use:
```bash
docker run -v $(pwd):/workspace/project --gpus all -it --rm <project_name>
```
Uncomment Apex in Dockerfile for mixed-precision support.
<br><br><br>



## ❤️&nbsp;&nbsp;Contributions
Have a question? Found a bug? Missing a specific feature? Ran into a problem? Feel free to file a new issue or PR with respective title and description. If you already found a solution to your problem, don't hesitate to share it. Suggestions for new best practices and tricks are always welcome!
<br><br><br><br>



## ℹ️&nbsp;&nbsp;Guide

### How To Get Started
- First, you should probably get familiar with [PyTorch Lightning](https://www.pytorchlightning.ai)
- Next, go through [Hydra quick start guide](https://hydra.cc/docs/intro/) and [basic Hydra tutorial](https://hydra.cc/docs/tutorials/basic/your_first_app/simple_cli/)
<br>


### How it works
By design, every run is initialized by [run.py](run.py) file. All PyTorch Lightning modules are dynamically instantiated from module paths specified in config. Example model config:
```yaml
_target_: src.models.mnist_model.MNISTLitModel
input_size: 784
lin1_size: 256
lin2_size: 256
lin3_size: 256
output_size: 10
lr: 0.001
```
Using this config we can instantiate the object with the following line:
```python
model = hydra.utils.instantiate(config.model)
```
This allows you to easily iterate over new models!<br>
Every time you create a new one, just specify its module path and parameters in appriopriate config file. <br>
The whole pipeline managing the instantiation logic is placed in [src/train.py](src/train.py).

<br>


### Main Project Configuration
Location: [configs/config.yaml](configs/config.yaml)<br>
Main project config contains default training configuration.<br>
It determines how config is composed when simply executing command `python run.py`.<br>
It also specifies everything that shouldn't be managed by experiment configurations.
<details>
<summary><b>Show main project configuration</b></summary>

```yaml
# specify here default training configuration
defaults:
    - trainer: default.yaml
    - model: mnist_model.yaml
    - datamodule: mnist_datamodule.yaml
    - callbacks: default.yaml  # set this to null if you don't want to use callbacks
    - logger: null  # set logger here or use command line (e.g. `python run.py logger=wandb`)

    - hydra: default.yaml

    - experiment: null
    - hparams_search: null


# path to original working directory
# hydra hijacks working directory by changing it to the current log directory,
# so it's useful to have this path as a special variable
# learn more here: https://hydra.cc/docs/next/tutorials/basic/running_your_app/working_directory
work_dir: ${hydra:runtime.cwd}


# path to folder with data
data_dir: ${work_dir}/data/


# pretty print config at the start of the run using Rich library
print_config: True


# disable python warnings if they annoy you
ignore_warnings: True
```

</details>
<br>


### Experiment Configuration
Location: [configs/experiment](configs/experiment)<br>
You should store all your experiment configurations in this folder.<br>
Experiment configurations allow you to overwrite parameters from main project configuration.

**Simple example**
```yaml
# to execute this experiment run:
# python run.py experiment=example_simple

defaults:
    - override /trainer: default.yaml
    - override /model: mnist_model.yaml
    - override /datamodule: mnist_datamodule.yaml
    - override /callbacks: default.yaml
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
    lr: 0.005

datamodule:
    train_val_test_split: [55_000, 5_000, 10_000]
    batch_size: 64
```
</details>


<details>
<summary><b>Show advanced example</b></summary>

```yaml
# to execute this experiment run:
# python run.py experiment=example_full

defaults:
    - override /trainer: null
    - override /model: null
    - override /datamodule: null
    - override /callbacks: null
    - override /logger: null

# we override default configurations with nulls to prevent them from loading at all
# instead we define all modules and their paths directly in this config,
# so everything is stored in one place

seed: 12345

trainer:
    _target_: pytorch_lightning.Trainer
    gpus: 0
    min_epochs: 1
    max_epochs: 10
    gradient_clip_val: 0.5

model:
    _target_: src.models.mnist_model.MNISTLitModel
    lr: 0.001
    weight_decay: 0.00005
    input_size: 784
    lin1_size: 256
    lin2_size: 256
    lin3_size: 128
    output_size: 10

datamodule:
    _target_: src.datamodules.mnist_datamodule.MNISTDataModule
    data_dir: ${data_dir}
    train_val_test_split: [55_000, 5_000, 10_000]
    batch_size: 64
    num_workers: 0
    pin_memory: False

logger:
    wandb:
        _target_: pytorch_lightning.loggers.wandb.WandbLogger
        project: "lightning-hydra-template"
        tags: ["best_model", "uwu"]
        notes: "Description of this model."
```

</details>

<br>

### Workflow
1. Write your PyTorch Lightning model (see [mnist_model.py](src/models/mnist_model.py) for example)
2. Write your PyTorch Lightning datamodule (see [mnist_datamodule.py](src/datamodules/mnist_datamodule.py) for example)
3. Write your experiment config, containing paths to your model and datamodule
4. Run training with chosen experiment config: `python run.py experiment=experiment_name`
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

You can change this structure by modifying paths in [hydra configuration](configs/hydra/default.yaml).
<br><br>


### Experiment Tracking
PyTorch Lightning supports the most popular logging frameworks:<br>
**[Weights&Biases](https://www.wandb.com/) · [Neptune](https://neptune.ai/) · [Comet](https://www.comet.ml/) · [MLFlow](https://mlflow.org) · [Aim](https://github.com/aimhubio/aim) · [Tensorboard](https://www.tensorflow.org/tensorboard/)**

These tools help you keep track of hyperparameters and output metrics and allow you to compare and visualize results. To use one of them simply complete its configuration in [configs/logger](configs/logger) and run:
 ```yaml
 python run.py logger=logger_name
 ```
You can use many of them at once (see [configs/logger/many_loggers.yaml](configs/logger/many_loggers.yaml) for example).<br>
You can also write your own logger.<br>
Lightning provides convenient method for logging custom metrics from inside LightningModule. Read the docs [here](https://pytorch-lightning.readthedocs.io/en/latest/extensions/logging.html#automatic-logging) or take a look at [MNIST example](src/models/mnist_model.py).
<br><br>



### Hyperparameter Search
Defining hyperparameter optimization is as easy as adding new config file to [configs/hparams_search](configs/hparams_search).
<details>
<summary><b>Show example</b></summary>

```yaml
defaults:
    - override /hydra/sweeper: optuna


# choose metric which will be optimized by Optuna
optimized_metric: "val/acc"


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
            model.lin1_size:
                type: categorical
                choices: [32, 64, 128, 256, 512]
            model.lin2_size:
                type: categorical
                choices: [32, 64, 128, 256, 512]
            model.lin3_size:
                type: categorical
                choices: [32, 64, 128, 256, 512]
```

</details>

Next, you can execute it with: `python run.py -m hparams_search=mnist_optuna`<br>
Using this approach doesn't require you to add any boilerplate into your pipeline, everything is defined in a single config file. You can use different optimization frameworks integrated with Hydra, like Optuna, Ax or Nevergrad.
<br><br>



### Inference
The following is example of loading model from checkpoint and running predictions.<br>
<details>
<summary><b>Show inference example</b></summary>

```python
from PIL import Image
from torchvision import transforms

from src.models.mnist_model import MNISTLitModel


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
    trained_model = MNISTLitModel.load_from_checkpoint(checkpoint_path=CKPT_PATH)

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
Template comes with example tests implemented with pytest library. <br>
To execute them simply run:
```yaml
# run all tests
pytest

# run tests from specific file
pytest tests/shell/test_basic_commands.py

# run all tests except the ones marked as slow
pytest -k "not slow"
```
I often find myself running into bugs that come out only in edge cases or on some specific hardware/environment. To speed up the development, I usually constantly execute tests that run a couple of quick 1 epoch experiments, like overfitting to 10 batches, training on 25% of data, etc. Those kind of tests don't check for any specific output - they exist to simply verify that executing some commands doesn't end up in throwing exceptions. You can find them implemented in [tests/shell](tests/shell) folder.

You can easily modify the commands in the scripts for your use case. If even 1 epoch is too much for your model, then you can make it run for a couple of batches instead (by using the right trainer flags).
<br><br>



### Callbacks
Template contains example callbacks enabling better Weights&Biases integration, which you can use as a reference for writing your own callbacks (see [wandb_callbacks.py](src/callbacks/wandb_callbacks.py)).<br>
To support reproducibility:
- **WatchModel**
- **UploadCodeAsArtifact**
- **UploadCheckpointsAsArtifact**

To provide examples of logging custom visualisations with callbacks only:
- **LogConfusionMatrix**
- **LogF1PrecRecHeatmap**
- **LogImagePredictions**

To see the result of all the callbacks attached, take a look at [this experiment dashboard](https://wandb.ai/hobglob/template-tests/runs/3rw7q70h).
<br><br>



### Multi-GPU Training
Lightning supports multiple ways of doing distributed training.<br>
The most common one is DDP, which spawns separate process for each GPU and averages gradients between them. To learn about other approaches read [lightning docs](https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html).

You can run DDP on mnist example with 4 GPUs like this:
```yaml
python run.py trainer.gpus=4 +trainer.accelerator="ddp"
```
⚠️ When using DDP you have to be careful how you write your models - learn more [here](https://pytorch-lightning.readthedocs.io/en/latest/advanced/multi_gpu.html).
<br><br>


### Extra Features
List of extra utilities available in the template:
- loading environment variables from [.env](.env.example) file
- pretty printing config with [Rich](https://github.com/willmcgugan/rich) library
- disabling python warnings
- debug mode
<!-- - (TODO) resuming latest run -->

You can easily remove any of those by modifying [run.py](run.py) and [src/train.py](src/train.py).
<br><br>

<!--
### Limitations
(TODO)
<br><br><br>
 -->


## Best Practices
<!--<details>
<summary><b>Use Docker</b></summary>

Docker makes it easy to initialize the whole training environment, e.g. when you want to execute experiments in cloud or on some private computing cluster. You can extend [dockerfiles](https://github.com/ashleve/lightning-hydra-template/tree/dockerfiles) provided in the template with your own instructions for building the image.<br>

</details> -->

<details>
<summary><b>Use Miniconda for GPU environments</b></summary>

Use miniconda for your python environments (it's usually unnecessary to install full anaconda environment, miniconda should be enough).
It makes it easier to install some dependencies, like cudatoolkit for GPU support.<br>
Example installation:
```yaml
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```
Create environment using bash script provided in the template:
```yaml
bash bash/setup_conda.sh
```

</details>

<details>
<summary><b>Use automatic code formatting</b></summary>

Use pre-commit hooks to standardize code formatting of your project and save mental energy.<br>
Simply install pre-commit package with:
```yaml
pip install pre-commit
```
Next, install hooks from [.pre-commit-config.yaml](.pre-commit-config.yaml):
```yaml
pre-commit install
```
After that your code will be automatically reformatted on every new commit.<br>
Currently template contains configurations of **black** (python code formatting), **isort** (python import sorting), **flake8** (python code analysis) and **prettier** (yaml formating). <br>

To reformat all files in the project use command:
```yaml
pre-commit run -a
```

</details>

<details>
<summary><b>Set private environment variables in .env file</b></summary>

System specific variables (e.g. absolute paths to datasets) should not be under version control or it will result in conflict between different users. Your private keys also shouldn't be versioned since you don't want them to be leaked.<br>

Template contains `.env.example` file, which serves as an example. Create a new file called `.env` (this name is excluded from version control in .gitignore).
You should use it for storing environment variables like this:
```yaml
MY_VAR=/home/user/my_system_path
```
All variables from `.env` are loaded in `run.py` automatically.

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
```yaml
dvc init
```
To start tracking a file or directory, use `dvc add`:
```yaml
dvc add data/MNIST
```
DVC stores information about the added file (or a directory) in a special .dvc file named data/MNIST.dvc, a small text file with a human-readable format. This file can be easily versioned like source code with Git, as a placeholder for the original data:
```yaml
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
    name="src",  # you should change "src" to your project name
    version="0.0.0",
    description="Describe Your Cool Project",
    author="",
    author_email="",
    # replace with your own github project link
    url="https://github.com/ashleve/lightning-hydra-template",
    install_requires=["pytorch-lightning>=1.2.0", "hydra-core>=1.0.6"],
    packages=find_packages(),
)
```
Now your project can be installed from local files:
```yaml
pip install -e .
```
Or directly from git repository:
```yaml
pip install git+git://github.com/YourGithubName/your-repo-name.git --upgrade
```
So any file can be easily imported into any other file like so:
```python
from project_name.models.mnist_model import MNISTLitModel
from project_name.datamodules.mnist_datamodule import MNISTDataModule
```

</details>

<!-- <details>
<summary><b>Make notebooks independent from other files</b></summary>

It's a good practice for jupyter notebooks to be portable. Try to make them independent from src files. If you need to access external code, try to embed it inside the notebook.

</details> -->
<br>




## Tricks
<details>
<summary><b>Automatic activation of virtual environment and tab completion when entering folder</b></summary>


Create a new file called `.autoenv` (this name is excluded from version control in `.gitignore`). <br>
You can use it to automatically execute shell commands when entering folder.

To setup this automation for bash, execute the following line:
```bash
echo "autoenv() { if [ -x .autoenv ]; then source .autoenv ; echo '.autoenv executed' ; fi } ; cd() { builtin cd \"\$@\" ; autoenv ; } ; autoenv" >> ~/.bashrc
```

Now you can add any commands to your `.autoenv` file, e.g. activation of virtual environment and hydra tab completion:
```bash
# activate conda environment
conda activate myenv

# activate hydra tab completion for bash
eval "$(python run.py -sc install=bash)"

# enable aliases for debugging
alias test='pytest'
alias debug1='python run.py debug=true'
alias debug2='python run.py trainer.gpus=1 trainer.max_epochs=1'
alias debug3='python run.py trainer.gpus=1 trainer.max_epochs=1 +trainer.limit_train_batches=0.1'
alias debug_wandb='python run.py trainer.gpus=1 trainer.max_epochs=1 logger=wandb logger.wandb.project=tests'
```
(these commands will be executed whenever you're openning or switching terminal to folder containing `.autoenv` file)

Lastly add execution previliges to your `.autoenv` file:
```
chmod +x .autoenv
```

**Explanation**<br>
The mentioned line appends your `.bashrc` file with 2 commands:
1. `autoenv() { if [ -x .autoenv ]; then source .autoenv ; echo '.autoenv executed' ; fi }` - this declares the `autoenv()` function, which executes `.autoenv` file if it exists in current work dir and has execution previligies
2. `cd() { builtin cd \"\$@\" ; autoenv ; } ; autoenv` - this extends behaviour of `cd` command, to make it execute `autoenv()` function each time you change folder in terminal or open new terminal

</details>


</details>


<details>
<summary><b>Accessing datamodule attributes in model</b></summary>

The simplest way is to pass datamodule attribute directly to model on initialization:
```python
datamodule = hydra.utils.instantiate(config.datamodule)

model = hydra.utils.instantiate(config.model, some_param=datamodule.some_param)
```
This is not a robust solution, since it assumes all your datamodules have `some_param` attribute available (otherwise the run will crash).
A better solution is to add Omegaconf resolver to your datamodule:
```python
from omegaconf import OmegaConf

# you can place this snippet in your datamodule __init__()
resolver_name = "datamodule"
OmegaConf.register_new_resolver(
    resolver_name,
    lambda name: getattr(self, name),
    use_cache=False
)
```
This way you can reference any datamodule attribute from your config like this:
```yaml
# this will get 'datamodule.some_param' field
some_parameter: ${datamodule: some_param}
```
When later accessing this field, say in your lightning model, it will get automatically resolved based on all resolvers that are registered. Remember not to access this field before datamodule is initialized. **You also need to set resolve to false in print_config() in [run.py](run.py) method or it will throw errors!**
```python
utils.print_config(config, resolve=False)
```

</details>

<!--
PrettyErrors and Rich exception handling,
 -->
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

<!-- <details>
<summary><b>List of repositories using this template</b></summary>

- [ashleve/graph_classification](https://github.com/ashleve/graph_classification) - benchmarking graph neural network architectures on graph classification datasets (Open Graph Benchmarks and image classification from superpixels)

</details> -->


<!-- ## :star:&nbsp; Stargazers Over Time

[![Stargazers over time](https://starchart.cc/ashleve/lightning-hydra-template.svg)](https://starchart.cc/ashleve/lightning-hydra-template) -->



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
```yaml
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
bash bash/setup_conda.sh

# install requirements
pip install -r requirements.txt
```

Train model with default configuration
```yaml
# default
python run.py

# train on CPU
python run.py trainer.gpus=0

# train on GPU
python run.py trainer.gpus=1
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)
```yaml
python run.py experiment=experiment_name
```

You can override any parameter from command line like this
```yaml
python run.py trainer.max_epochs=20 datamodule.batch_size=64
```

<br>
