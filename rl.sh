#!/bin/bash

python train_human_rl.py --checkpoint data/gomoku_checkpoints/final \
    	                 --epochs-rm 20 \
			 --lr-rm 1e-4 \
			 --epochs-ppo 5 \
		         --ppo-steps 50 \
			 --lr-ppo 1e-6 \
			 --ppo-inner-epochs 4 \
			 --ppo-clip 0.1 \
			 --ppo-entropy-coef 0.01 \
			 --ppo-value-coef 0.5 \
			 --batch-size 32 \
			 --device cuda
