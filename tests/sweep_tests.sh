# TESTS FOR HYPERPARAMETER SWEEPS
# TO EXECUTE:
# bash tests/sweep_tests.sh

# Test default hydra sweep with wandb logging
echo TEST 1
python train.py --multirun \
+experiment=exp_example_simple \
trainer.gpus=1 trainer.max_epochs=2 seed=12345 \
datamodule.batch_size=32,64,128 model.lr=0.001,0.003 \
datamodule.num_workers=12 datamodule.pin_memory=True print_config=False \
logger=wandb logger.wandb.project="env_tests" logger.wandb.group="HydraSweep_MNIST_SimpleDenseNet_seed12345"

# Test optuna sweep with wandb logging
echo TEST 2
python train.py --multirun --config-name config_optuna.yaml \
+experiment=exp_example_simple \
trainer.gpus=1 trainer.max_epochs=5 seed=12345 \
datamodule.num_workers=12 datamodule.pin_memory=True \
logger=wandb logger.wandb.project="env_tests" logger.wandb.group="Optuna_MNIST_SimpleDenseNet_seed12345"
