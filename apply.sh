#!/bin/bash

python apply_human_rl.py --source data/gomoku_checkpoints/ \
	                 --human-rl data/human_rl/best_model.pt \
			 --output data/gomoku_checkpoints
