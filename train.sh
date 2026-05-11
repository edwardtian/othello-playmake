#!/bin/bash

sudo nvidia-smi -i 0 -pl 300
sudo nvidia-smi -i 1 -pl 250
python train_fast.py --steps 500000 --workers 32 --resume
# CUDA_VISIBLE_DEVICES=0 python train_fast.py --steps 100000 --workers 16


