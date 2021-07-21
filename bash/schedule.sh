#!/bin/bash
# Shedule execution of many runs
# Run from root folder with: bash bash/schedule.sh

python run.py trainer.max_epochs=5

python run.py trainer.max_epochs=5 experiment=example_simple

python run.py trainer.max_epochs=5 experiment=example_full
