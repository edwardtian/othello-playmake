#!/bin/bash
# Behavioral Cloning — stable alternative to PPO
# After PPO repeatedly failed to learn (frozen policy / entropy collapse),
# BC directly trains the policy on human-preferred moves via cross-entropy.

python train_human_rl.py \
    --checkpoint data/gomoku_checkpoints/final \
    --epochs-rm 20 \
    --lr-rm 1e-4 \
    --epochs-ppo 30 \
    --lr-ppo 1e-4 \
    --batch-size 64 \
    --use-bc \
    --device cuda
