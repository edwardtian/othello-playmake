#!/bin/bash
# Conservative but effective Human RL tuning
# For ~2,700 preferences

python train_human_rl.py \
    --checkpoint data/gomoku_checkpoints/final \
    --epochs-rm 20 \
    --lr-rm 1e-4 \
    --epochs-ppo 10 \
    --ppo-steps 100 \
    --lr-ppo 3e-5 \
    --ppo-inner-epochs 4 \
    --ppo-clip 0.15 \
    --ppo-entropy-coef 0.02 \
    --ppo-value-coef 0.3 \
    --batch-size 64 \
    --original-value-weight 0.2 \
    --device cuda
