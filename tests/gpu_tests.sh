# !/bin/bash
# Quick tests for gpu

# To execute:
# bash tests/gpu_tests.sh

# Method for printing test name
echo() {
  termwidth="$(tput cols)"
  padding="$(printf '%0.1s' ={1..500})"
  printf '\e[33m%*.*s %s %*.*s\n\e[0m' 0 "$(((termwidth-2-${#1})/2))" "$padding" "$1" 0 "$(((termwidth-1-${#1})/2))" "$padding"
}

# Make python hide warnings
export PYTHONWARNINGS="ignore"


echo "TEST 1"
echo "Train on GPU (2 epochs)"
python run.py trainer.gpus=-1 trainer.max_epochs=2 \
print_config=false

echo "TEST 2"
echo "Train with 4 workers and cuda pinned memory (2 epochs)"
python run.py trainer.gpus=-1 trainer.max_epochs=2 \
datamodule.num_workers=4 datamodule.pin_memory=True \
print_config=false

echo "TEST 3"
echo "Train with mixed precision (apex, amp level 01, 1 epoch)"
python run.py trainer.gpus=-1 trainer.max_epochs=1 \
+trainer.amp_backend="apex" +trainer.amp_level="O1" +trainer.precision=16 \
print_config=false

echo "TEST 4"
echo "Train with mixed precision (apex, amp level 02, 1 epoch)"
python run.py trainer.gpus=-1 trainer.max_epochs=1 \
+trainer.amp_backend="apex" +trainer.amp_level="O2" +trainer.precision=16 \
print_config=false

echo "TEST 5"
echo "Train with mixed precision (apex, amp level 03, 1 epoch)"
python run.py trainer.gpus=-1 trainer.max_epochs=1 \
+trainer.amp_backend="apex" +trainer.amp_level="O3" +trainer.precision=16 \
print_config=false
