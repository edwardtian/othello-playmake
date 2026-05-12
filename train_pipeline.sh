#!/bin/bash
# Complete workflow: Human-vs-human -> BC training -> Self-play

set -e

echo "=========================================="
echo "Human-to-Self-Play Training Pipeline"
echo "=========================================="
echo ""

# Step 1: Check for human games
GAMES_DIR="data/human_games"
GAME_COUNT=$(find "$GAMES_DIR" -name "*.json" 2>/dev/null | wc -l)

echo "Found $GAME_COUNT human game(s) in $GAMES_DIR"
echo ""

if [ "$GAME_COUNT" -eq 0 ]; then
    echo "ERROR: No human games found."
    echo ""
    echo "To record human games:"
    echo "  1. Start web server:"
    echo "     python -m uvicorn web.gomoku_server:app --host 0.0.0.0 --port 8080"
    echo "  2. Select 'Human vs Human (record for training)' mode"
    echo "  3. Play games and click 'Submit Game' after each game"
    echo ""
    exit 1
fi

# Step 2: Train initial model from human games
echo "Step 1: Training initial model from human games..."
python train_from_human.py \
    --games "$GAMES_DIR" \
    --epochs 50 \
    --batch-size 64 \
    --lr 1e-3 \
    --device cuda

echo ""
echo "Step 2: Initial model saved to data/human_init/"
echo ""

# Step 3: Start self-play training from the human-init checkpoint
echo "Step 3: Starting self-play training from human-init checkpoint..."
python train_gomoku.py \
    --checkpoint data/human_init/checkpoint \
    --steps 100000 \
    --workers 128 \
    --fp16 \
    --resume
