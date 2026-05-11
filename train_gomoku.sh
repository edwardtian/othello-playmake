#!/bin/bash

sudo nvidia-smi -i 0 -pl 300
sudo nvidia-smi -i 1 -pl 250
python train_gomoku.py --steps 150000 --workers 64 --fp16 --resume --eval-parallel 16

