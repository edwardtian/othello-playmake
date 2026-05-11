#!/bin/bash

python train_human_rl.py --checkpoint data/gomoku_checkpoints/final \
    	                 --epochs-rm 20 \
			 --epochs-ppo 10 \
			 --batch-size 64 \
			 --device cuda
