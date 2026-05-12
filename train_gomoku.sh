#!/bin/bash

sudo nvidia-smi -i 0 -pl 300
sudo nvidia-smi -i 1 -pl 300
python train_gomoku.py --steps 100000 --workers 64 --fp16 --eval-parallel 16

