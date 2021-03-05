# THESE ARE JUST A COUPLE OF QUICK EXPERIMENTS TO TEST IF YOUR MODEL DOESN'T CRASH UNDER DIFFERENT CONDITIONS
# TO EXECUTE:
# bash tests/quick_tests.sh

# conda activate testenv

export PYTHONWARNINGS="ignore"

print_test_name() {
  termwidth="$(tput cols)"
  padding="$(printf '%0.1s' ={1..500})"
  printf '\e[33m%*.*s %s %*.*s\n\e[0m' 0 "$(((termwidth-2-${#1})/2))" "$padding" "$1" 0 "$(((termwidth-1-${#1})/2))" "$padding"
}


# Test for CPU
print_test_name "TEST 1"
python train.py trainer.gpus=0 trainer.max_epochs=1 print_config=false

# Test for GPU
print_test_name "TEST 2"
python train.py trainer.gpus=1 trainer.max_epochs=1 print_config=false

# Test multiple workers and cuda pinned memory
print_test_name "TEST 3"
python train.py trainer.gpus=1 trainer.max_epochs=2 print_config=false\
datamodule.num_workers=4 datamodule.pin_memory=True  

# Test all experiment configs
print_test_name "TEST 4"
python train.py -m '+experiment=glob(*)' trainer.gpus=1 trainer.max_epochs=3 print_config=false

# Test with debug trainer
print_test_name "TEST 5"
python train.py trainer=debug_trainer print_config=false

# Overfit to 10 bathes
print_test_name "TEST 6"
python train.py trainer.min_epochs=20 trainer.max_epochs=20 +trainer.overfit_batches=10 print_config=false

# Test default hydra sweep over hyperparameters (runs 4 different combinations for 1 epoch)
print_test_name "TEST 7"
python train.py -m datamodule.batch_size=32,64 model.lr=0.001,0.003 print_config=false \
trainer.gpus=1 trainer.max_epochs=1
