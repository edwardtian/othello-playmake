"""
Human Game Parser — convert recorded human-vs-human games into training data.

Each game produces a list of (state, policy_target, value_target) tuples:
    - state: (3, 15, 15) board state before the move
    - policy_target: one-hot at the action taken
    - value_target: +1 if that player eventually won, -1 if lost, 0 for draw
"""

import os
import json
import glob
import numpy as np
from typing import List, Tuple


def load_human_games(games_dir: str = 'data/human_games') -> List[dict]:
    """Load all recorded human games from disk."""
    games = []
    pattern = os.path.join(games_dir, '**/*.json')
    for path in glob.glob(pattern, recursive=True):
        try:
            with open(path, 'r') as f:
                game = json.load(f)
            if len(game.get('moves', [])) >= 5:
                games.append(game)
        except Exception as e:
            print(f"[Warning] Failed to load {path}: {e}")
    return games


def game_to_training_data(game: dict, action_size: int = 225) -> List[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Convert a single human game into training tuples.

    Args:
        game: Dict with 'moves' list and 'winner'
        action_size: Number of possible actions (225 for 15x15 Gomoku)

    Returns:
        List of (state, policy_target, value_target) tuples
    """
    winner = game.get('winner', 0)
    moves = game.get('moves', [])
    training_data = []

    for move in moves:
        player = move['player']
        action = move['action']
        state = np.array(move['state_before'], dtype=np.float32)

        # Policy target: one-hot at the played action
        policy = np.zeros(action_size, dtype=np.float32)
        policy[action] = 1.0

        # Value target: game outcome from this player's perspective
        if winner == 0:
            value = 0.0
        elif winner == player:
            value = 1.0
        else:
            value = -1.0

        training_data.append((state, policy, value))

    return training_data


def load_all_training_data(games_dir: str = 'data/human_games', action_size: int = 225):
    """
    Load all human games and convert to training data.

    Returns:
        List of (state, policy, value) tuples
    """
    games = load_human_games(games_dir)
    all_data = []
    for game in games:
        all_data.extend(game_to_training_data(game, action_size))
    return all_data, len(games)
