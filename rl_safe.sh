#!/bin/bash
# Post-collapse conservative recovery script
# After seeing entropy crash from 4.1 -> 0.25 in 5 epochs,
# this uses aggressive entropy protection + early stopping.

python train_human_rl.py \
    --checkpoint data/gomoku_checkpoints/final \
    --epochs-rm 20 \
    --lr-rm 1e-4 \
    --epochs-ppo 6 \
    --ppo-steps 80 \
    --lr-ppo 2e-5 \
    --ppo-inner-epochs 3 \
    --ppo-clip 0.1 \
    --ppo-entropy-coef 0.05 \
    --ppo-value-coef 0.2 \
    --ppo-entropy-min 1.5 \
    --batch-size 64 \
    --original-value-weight 0.3 \
    --device cuda
