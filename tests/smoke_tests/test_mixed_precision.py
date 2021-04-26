# echo "TEST 3"
# echo "Train with mixed precision (apex, amp level 01, 1 epoch)"
# python run.py trainer.gpus=-1 trainer.max_epochs=1 \
# +trainer.amp_backend="apex" +trainer.amp_level="O1" +trainer.precision=16 \
# print_config=false

# echo "TEST 4"
# echo "Train with mixed precision (apex, amp level 02, 1 epoch)"
# python run.py trainer.gpus=-1 trainer.max_epochs=1 \
# +trainer.amp_backend="apex" +trainer.amp_level="O2" +trainer.precision=16 \
# print_config=false

# echo "TEST 5"
# echo "Train with mixed precision (apex, amp level 03, 1 epoch)"
# python run.py trainer.gpus=-1 trainer.max_epochs=1 \
# +trainer.amp_backend="apex" +trainer.amp_level="O3" +trainer.precision=16 \
# print_config=false
