"""
Play Gomoku against the AI.

Usage:
    python play_gomoku.py                    # Play with default (untrained) model
    python play_gomoku.py --checkpoint data/gomoku_checkpoints/best_model.pt
    python play_gomoku.py --checkpoint data/gomoku_checkpoints/checkpoint_10000/model.pt --human-white
    python play_gomoku.py --sims 800         # Stronger AI (slower)
"""

import sys
import os
import argparse
import numpy as np
import torch

from game.gomoku import GomokuGame, BLACK, WHITE
from ai.model import create_model
from ai.mcts_batched import BatchedMCTS


def print_board(game: GomokuGame, show_legal: bool = True):
    """Print a nice Gomoku board."""
    board = game.board
    size = game.board_size
    legal = set(game.get_legal_moves()) if show_legal else set()

    # Column headers
    header = "    " + " ".join(f"{i:2d}" for i in range(size))
    print(header)
    print("   +" + "---" * size + "+")

    for row in range(size):
        row_str = f"{row:2d} |"
        for col in range(size):
            action = row * size + col
            val = board[row, col]
            if val == BLACK:
                row_str += " X "
            elif val == WHITE:
                row_str += " O "
            elif action in legal and show_legal:
                row_str += " . "
            else:
                row_str += "   "
        row_str += "|"
        print(row_str)

    print("   +" + "---" * size + "+")

    # Show scores
    bc, wc = game.count_pieces()
    print(f"   Black: {bc}  White: {wc}")
    if not game.is_game_over():
        print(f"   Turn: {'Black (X)' if game.current_player == BLACK else 'White (O)'}")


def get_human_move(game: GomokuGame) -> int:
    """Get a move from the human player."""
    legal = game.get_legal_moves()
    size = game.board_size

    while True:
        try:
            move_str = input(f"\nYour move (row col, e.g. '7 7', or 'q' to quit): ").strip()
            if move_str.lower() == 'q':
                return -1

            parts = move_str.split()
            if len(parts) != 2:
                print("Please enter row and column separated by space.")
                continue

            row, col = int(parts[0]), int(parts[1])
            action = row * size + col

            if action not in legal:
                print(f"Invalid move. Legal moves include: {[(m // size, m % size) for m in legal[:10]]}...")
                continue

            return action
        except ValueError:
            print("Invalid input. Please enter two numbers.")


def main():
    parser = argparse.ArgumentParser(description='Play Gomoku against AI')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint')
    parser.add_argument('--human-color', type=str, default='black', choices=['black', 'white'],
                        help='Which color you play (black goes first)')
    parser.add_argument('--sims', type=int, default=400, help='MCTS simulations per AI move')
    parser.add_argument('--device', type=str, default='auto', help='Device (auto/cpu/cuda)')
    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    # Model config for Gomoku
    board_size = 15
    action_size = 225
    model_config = {
        'num_blocks': 10,
        'num_channels': 128,
        'board_size': board_size,
        'action_size': action_size,
    }

    # Load model
    print("Loading model...")
    model = create_model(**model_config, device=device)

    if args.checkpoint and os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("Using untrained model (random play)")

    model.eval()

    # Create MCTS
    mcts = BatchedMCTS(model, num_simulations=args.sims, batch_size=16, action_size=action_size)

    # Determine who plays which color
    human_player = BLACK if args.human_color == 'black' else WHITE
    ai_player = WHITE if human_player == BLACK else BLACK

    print("\n" + "=" * 50)
    print("        GOMOKU - Play against AI")
    print("=" * 50)
    print(f"Board: 15x15")
    print(f"You play: {'Black (X)' if human_player == BLACK else 'White (O)'}")
    print(f"AI plays: {'Black (X)' if ai_player == BLACK else 'White (O)'}")
    print(f"AI MCTS simulations: {args.sims}")
    print(f"Device: {device}")
    print("Enter moves as: row col  (e.g. '7 7' for center)")
    print("Enter 'q' to quit")
    print("=" * 50)

    game = GomokuGame()
    print_board(game)

    while not game.is_game_over():
        current = game.current_player

        if current == human_player:
            action = get_human_move(game)
            if action == -1:
                print("Game quit.")
                return
        else:
            # AI move
            print("\nAI is thinking...")
            import time as _time
            t0 = _time.time()
            action_probs, value = mcts.search(game, temperature=0.0)
            action = int(np.argmax(action_probs))
            elapsed = _time.time() - t0
            print(f"AI chose ({action // board_size}, {action % board_size}) in {elapsed:.1f}s")
            print(f"AI position value: {value[0]:.3f}")

        success, msg = game.make_move(action)
        if not success:
            print(f"Move failed: {msg}")
            continue

        print(f"\n{msg}")
        print_board(game)

    # Game over
    winner, bc, wc = game.get_winner()
    print("\n" + "=" * 50)
    if winner == 0:
        print("Game Over - DRAW!")
    elif winner == human_player:
        print("Game Over - YOU WIN! Congratulations!")
    else:
        print("Game Over - AI WINS!")
    print(f"Final score - Black: {bc}, White: {wc}")
    print("=" * 50)


if __name__ == '__main__':
    main()
