"""
Evaluation arena for Othello AI.

Pits a challenger model against the current champion.
If the challenger wins >55% of games, it becomes the new champion.
"""

import os
import json
import numpy as np
import torch
from typing import Dict, Tuple
from game.othello import OthelloGame, BLACK, WHITE
from ai.model import OthelloNet
from ai.mcts import MCTS


def play_match(
    model_a: OthelloNet,
    model_b: OthelloNet,
    num_games: int = 100,
    num_simulations: int = 400,
    temperature: float = 0.0,
    device: str = 'cpu',
) -> Dict[str, int]:
    """
    Play a match between two models.

    Args:
        model_a: First model
        model_b: Second model
        num_games: Number of games to play (each model plays both colors)
        num_simulations: MCTS simulations per move
        temperature: Temperature for move selection (0 = argmax)
        device: Device to run models on

    Returns:
        Dictionary with match results:
        {'model_a_wins': int, 'model_b_wins': int, 'draws': int}
    """
    mcts_a = MCTS(model_a, num_simulations=num_simulations)
    mcts_b = MCTS(model_b, num_simulations=num_simulations)

    results = {'model_a_wins': 0, 'model_b_wins': 0, 'draws': 0}

    # Play half the games with model_a as black, half with model_a as white
    for game_idx in range(num_games):
        game = OthelloGame()

        # Alternate colors
        if game_idx % 2 == 0:
            black_mcts, white_mcts = mcts_a, mcts_b
        else:
            black_mcts, white_mcts = mcts_b, mcts_a

        # Play game
        while not game.is_game_over():
            if game.current_player == BLACK:
                action = black_mcts.get_best_move(game, temperature=temperature)
            else:
                action = white_mcts.get_best_move(game, temperature=temperature)
            game.make_move(action)

        # Determine winner
        winner, _, _ = game.get_winner()

        if winner == 0:
            results['draws'] += 1
        elif game_idx % 2 == 0:
            # model_a was black
            if winner == BLACK:
                results['model_a_wins'] += 1
            else:
                results['model_b_wins'] += 1
        else:
            # model_a was white
            if winner == WHITE:
                results['model_a_wins'] += 1
            else:
                results['model_b_wins'] += 1

    return results


def evaluate_challenger(
    champion_model: OthelloNet,
    challenger_model: OthelloNet,
    num_games: int = 200,
    num_simulations: int = 400,
    win_threshold: float = 0.55,
    device: str = 'cpu',
) -> Tuple[bool, Dict]:
    """
    Evaluate if challenger should replace champion.

    Returns:
        (is_better, results_dict)
        is_better: True if challenger win rate > win_threshold
    """
    results = play_match(
        champion_model,
        challenger_model,
        num_games=num_games,
        num_simulations=num_simulations,
        device=device,
    )

    total = results['model_a_wins'] + results['model_b_wins'] + results['draws']
    challenger_wins = results['model_b_wins']
    challenger_win_rate = challenger_wins / total if total > 0 else 0.0

    is_better = challenger_win_rate > win_threshold

    detailed_results = {
        'challenger_win_rate': challenger_win_rate,
        'challenger_wins': challenger_wins,
        'champion_wins': results['model_a_wins'],
        'draws': results['draws'],
        'total_games': total,
        'is_better': is_better,
    }

    return is_better, detailed_results


def update_elo(
    champion_elo: float,
    challenger_elo: float,
    challenger_score: float,
    k_factor: float = 32.0,
) -> Tuple[float, float]:
    """
    Update ELO ratings after a match.

    Args:
        champion_elo: Current champion ELO
        challenger_elo: Current challenger ELO
        challenger_score: Score from challenger's perspective (1=win, 0.5=draw, 0=loss)
        k_factor: ELO K-factor

    Returns:
        (new_champion_elo, new_challenger_elo)
    """
    expected_challenger = 1.0 / (1.0 + 10.0 ** ((champion_elo - challenger_elo) / 400.0))

    new_challenger_elo = challenger_elo + k_factor * (challenger_score - expected_challenger)
    new_champion_elo = champion_elo + k_factor * ((1 - challenger_score) - (1 - expected_challenger))

    return new_champion_elo, new_challenger_elo


def load_champion(checkpoint_dir: str, model: OthelloNet, device: str = 'cpu') -> bool:
    """Load champion model if it exists."""
    champion_path = os.path.join(checkpoint_dir, 'best_model.pt')
    if os.path.exists(champion_path):
        state_dict = torch.load(champion_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        return True
    return False


def save_elo_history(elo_history: list, log_dir: str):
    """Save ELO history to JSON."""
    path = os.path.join(log_dir, 'elo_history.json')
    with open(path, 'w') as f:
        json.dump(elo_history, f, indent=2)
