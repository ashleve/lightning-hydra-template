# THESE ARE JUST A COUPLE OF QUICK RUN EXAMPLES 
# TO TEST IF YOUR MODEL WORKS UNDER DIFFERENT CONDITIONS

# Test for CPU and GPU
python train.py trainer.gpus=0 trainer.max_epochs=1 
python train.py trainer.gpus=1 trainer.max_epochs=1 

# Test multiple workers and cuda pinned memory
python train.py trainer.gpus=1 trainer.max_epochs=2 \
datamodule.num_workers=4 datamodule.pin_memory=True  

# Test all experiment configs
python train.py --multirun '+experiment=glob(*)' trainer.gpus=1 trainer.min_epochs=1 trainer.max_epochs=3

# Test with debug trainer
python train.py trainer=debug_trainer

# Overfit to 10 bathes
python train.py trainer.min_epochs=20 trainer.max_epochs=20 +trainer.overfit_batches=10

# Test default hydra sweep over hyperparameters (runs 4 different combinations)
python train.py --multirun datamodule.batch_size=32,64 model.lr=0.001,0.003 \
trainer.gpus=1 trainer.max_epochs=1
