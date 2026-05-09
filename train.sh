#!/bin/bash

nvidia-smi -i 0 -f 100
nvidia-smi -i 1 -f 100
sudo nvidia-smi -i 0 -pl 300
sudo nvidia-smi -i 1 -pl 250
python train_fast.py --steps 500000 --workers 32 --resume



