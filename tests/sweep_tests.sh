# Test default hydra sweep with wandb logging
python train.py --multirun \
+experiment=exp_example_simple \
trainer.gpus=1 trainer.max_epochs=2 seed=12345 \
datamodule.batch_size=32,64,128 model.lr=0.001,0.003 \
logger=wandb logger.wandb.group="HydraSweep_MNIST_SimpleDenseNet_seed12345"

# Test optuna sweep with wandb logging
python train.py --multirun --config-name config_optuna.yaml \
+experiment=exp_example_simple \
trainer.gpus=1 trainer.max_epochs=5 seed=12345 \
logger=wandb logger.wandb.group="Optuna_MNIST_SimpleDenseNet_seed12345"
