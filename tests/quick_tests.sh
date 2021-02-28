# THESE ARE JUST A COUPLE OF QUICK EXPERIMENTS TO TEST IF YOUR MODEL DOESN'T CRASH UNDER DIFFERENT CONDITIONS
# TO EXECUTE:
# bash tests/quick_tests.sh

# Test for CPU
echo TEST 1
python train.py trainer.gpus=0 trainer.max_epochs=1

# Test for GPU
echo TEST 2
python train.py trainer.gpus=1 trainer.max_epochs=1 

# Test multiple workers and cuda pinned memory
echo TEST 3 
python train.py trainer.gpus=1 trainer.max_epochs=2 \
datamodule.num_workers=4 datamodule.pin_memory=True  

# Test all experiment configs
echo TEST 4
python train.py --multirun '+experiment=glob(*)' trainer.gpus=1 trainer.min_epochs=1 trainer.max_epochs=3

# Test with debug trainer
echo TEST 5
python train.py trainer=debug_trainer

# Overfit to 10 bathes
echo TEST 6
python train.py trainer.min_epochs=20 trainer.max_epochs=20 +trainer.overfit_batches=10

# Test default hydra sweep over hyperparameters (runs 4 different combinations for 1 epoch)
echo TEST 7
python train.py --multirun datamodule.batch_size=32,64 model.lr=0.001,0.003 \
trainer.gpus=1 trainer.max_epochs=1
