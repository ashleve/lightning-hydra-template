# !/bin/bash
# Test hyperparameter sweeps

# To execute:
# bash tests/sweep_tests.sh

# Method for printing test name
echo() {
  termwidth="$(tput cols)"
  padding="$(printf '%0.1s' ={1..500})"
  printf '\e[33m%*.*s %s %*.*s\n\e[0m' 0 "$(((termwidth-2-${#1})/2))" "$padding" "$1" 0 "$(((termwidth-1-${#1})/2))" "$padding"
}

# Make python hide warnings
export PYTHONWARNINGS="ignore"


echo "TEST 1"
echo "Default hydra sweep with wandb logging"
python run.py -m \
+experiment=exp_example_simple \
datamodule.batch_size=64,128 optimizer.lr=0.001,0.003 \
trainer.gpus=-1 trainer.max_epochs=2 \
datamodule.num_workers=12 datamodule.pin_memory=True \
logger=wandb logger.wandb.project="env_tests" logger.wandb.group="DefaultSweep_MNIST_SimpleDenseNet"

echo "TEST 2"
echo "Optuna sweep with wandb logging"
python run.py -m --config-name config_optuna.yaml \
+experiment=exp_example_simple \
trainer.gpus=-1 trainer.max_epochs=5 \
datamodule.num_workers=12 datamodule.pin_memory=True \
logger=wandb logger.wandb.project="env_tests" logger.wandb.group="Optuna_MNIST_SimpleDenseNet"
