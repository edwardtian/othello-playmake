"""
Self-play game generation for Othello.

Generates training data by having the AI play against itself using MCTS.
Each game produces a list of (state, mcts_policy, action, outcome) tuples.
"""

import numpy as np
import torch
from typing import List, Tuple, Callable
from game.othello import OthelloGame, BLACK, WHITE
from ai.mcts import MCTS


def generate_self_play_game(
    mcts: MCTS,
    temperature_threshold: int = 30,
    temperature_init: float = 1.0,
    temperature_final: float = 0.0,
) -> List[Tuple[np.ndarray, np.ndarray, int, float]]:
    """
    Play one self-play game using MCTS.

    Args:
        mcts: MCTS instance with a neural network
        temperature_threshold: Number of moves before switching to argmax
        temperature_init: Temperature for first moves (exploration)
        temperature_final: Temperature after threshold (exploitation)

    Returns:
        List of (state_planes, mcts_policy, action, outcome) tuples.
        Outcome is from the perspective of the current player at that state:
        +1 = win, -1 = loss, 0 = draw
    """
    game = OthelloGame()
    game_history = []
    move_count = 0

    while not game.is_game_over():
        # Determine temperature
        if move_count < temperature_threshold:
            temperature = temperature_init
        else:
            temperature = temperature_final

        # Run MCTS
        action_probs, _ = mcts.search(game, temperature=temperature)

        # Sample action from policy
        action = np.random.choice(len(action_probs), p=action_probs)

        # Store state, policy, and actual action taken
        state = game.get_state_planes()
        game_history.append((state, action_probs.copy(), action, game.current_player))

        success, _ = game.make_move(action)
        if not success:
            raise RuntimeError(f"MCTS returned invalid move: {action}")

        move_count += 1

    # Determine winner
    winner, _, _ = game.get_winner()

    # Assign outcomes from perspective of each state's current player
    training_data = []
    for state, policy, action, current_player in game_history:
        if winner == 0:
            outcome = 0.0
        elif winner == current_player:
            outcome = 1.0
        else:
            outcome = -1.0
        training_data.append((state, policy, action, outcome))

    return training_data


def default_temperature_fn(move_count: int, threshold: int = 30) -> float:
    """Default temperature schedule."""
    return 1.0 if move_count < threshold else 0.0


def generate_self_play_games(
    mcts: MCTS,
    num_games: int,
    temperature_fn: Callable[[int], float] = None,
) -> List[List[Tuple[np.ndarray, np.ndarray, int, float]]]:
    """
    Generate multiple self-play games.

    Args:
        mcts: MCTS instance
        num_games: Number of games to play
        temperature_fn: Function move_count -> temperature

    Returns:
        List of game histories, each a list of (state, policy, action, outcome) tuples.
    """
    if temperature_fn is None:
        temperature_fn = default_temperature_fn

    games = []
    for _ in range(num_games):
        # Reuse MCTS but with fresh search each game
        game_data = generate_self_play_game(
            mcts,
            temperature_threshold=30,
            temperature_init=1.0,
            temperature_final=0.0,
        )
        games.append(game_data)

    return games
