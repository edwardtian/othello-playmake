"""
Evaluation arena for Othello AI.

Pits a challenger model against the current champion.
If the challenger wins >55% of games, it becomes the new champion.
"""

import os
import json
import numpy as np
import torch
import torch.multiprocessing as mp
from typing import Dict, Tuple
from game.othello import OthelloGame, BLACK, WHITE
from ai.model import OthelloNet
from ai.mcts_batched import BatchedMCTS


def _play_match_worker(args) -> Dict[str, int]:
    """
    Worker function for parallel match evaluation.
    Runs in a spawned process.
    """
    import torch
    from ai.model import create_model
    from ai.mcts_batched import BatchedMCTS

    (
        model_a_state,
        model_b_state,
        model_config,
        num_games,
        num_simulations,
        temperature,
        device,
        game_module,
    ) = args

    # Load models
    model_a = create_model(**model_config, device=device)
    model_b = create_model(**model_config, device=device)
    model_a.load_state_dict(model_a_state)
    model_b.load_state_dict(model_b_state)
    model_a.eval()
    model_b.eval()

    # Import game class
    if game_module == "gomoku":
        from game.gomoku import GomokuGame, BLACK, WHITE
        game_class = GomokuGame
    else:
        from game.othello import OthelloGame, BLACK, WHITE
        game_class = OthelloGame

    mcts_a = BatchedMCTS(
        model_a,
        num_simulations=num_simulations,
        batch_size=32,
        action_size=model_config["action_size"],
    )
    mcts_b = BatchedMCTS(
        model_b,
        num_simulations=num_simulations,
        batch_size=32,
        action_size=model_config["action_size"],
    )

    results = {"model_a_wins": 0, "model_b_wins": 0, "draws": 0}

    for game_idx in range(num_games):
        game = game_class()

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
            results["draws"] += 1
        elif game_idx % 2 == 0:
            # model_a was black
            if winner == BLACK:
                results["model_a_wins"] += 1
            else:
                results["model_b_wins"] += 1
        else:
            # model_a was white
            if winner == WHITE:
                results["model_a_wins"] += 1
            else:
                results["model_b_wins"] += 1

    return results


def _play_match_worker_queue(args, result_queue):
    """Wrapper that puts result into a queue for manual Process spawning."""
    result = _play_match_worker(args)
    result_queue.put(result)


def _extract_model_config(model: OthelloNet) -> Dict:
    """Extract model configuration from a model instance."""
    return {
        "num_blocks": getattr(model, "num_blocks", 20),
        "num_channels": getattr(model, "num_channels", 256),
        "board_size": getattr(model, "board_size", 8),
        "action_size": getattr(model, "action_size", 65),
    }


def play_match(
    model_a: OthelloNet,
    model_b: OthelloNet,
    num_games: int = 100,
    num_simulations: int = 400,
    temperature: float = 0.0,
    device: str = "cpu",
    game_class=None,
    num_parallel: int = 1,
) -> Dict[str, int]:
    """
    Play a match between two models using batched MCTS.

    Args:
        model_a: First model
        model_b: Second model
        num_games: Number of games to play (each model plays both colors)
        num_simulations: MCTS simulations per move
        temperature: Temperature for move selection (0 = argmax)
        device: Device to run models on
        game_class: Game class to instantiate (default: OthelloGame)
        num_parallel: Number of parallel processes for evaluation

    Returns:
        Dictionary with match results:
        {'model_a_wins': int, 'model_b_wins': int, 'draws': int}
    """
    if game_class is None:
        from game.othello import OthelloGame
        game_class = OthelloGame

    # Single-threaded fallback
    num_parallel = min(num_parallel, num_games)
    if num_parallel <= 1:
        mcts_a = BatchedMCTS(model_a, num_simulations=num_simulations, batch_size=32)
        mcts_b = BatchedMCTS(model_b, num_simulations=num_simulations, batch_size=32)

        results = {"model_a_wins": 0, "model_b_wins": 0, "draws": 0}

        for game_idx in range(num_games):
            game = game_class()

            if game_idx % 2 == 0:
                black_mcts, white_mcts = mcts_a, mcts_b
            else:
                black_mcts, white_mcts = mcts_b, mcts_a

            while not game.is_game_over():
                if game.current_player == BLACK:
                    action = black_mcts.get_best_move(game, temperature=temperature)
                else:
                    action = white_mcts.get_best_move(game, temperature=temperature)
                game.make_move(action)

            winner, _, _ = game.get_winner()

            if winner == 0:
                results["draws"] += 1
            elif game_idx % 2 == 0:
                if winner == BLACK:
                    results["model_a_wins"] += 1
                else:
                    results["model_b_wins"] += 1
            else:
                if winner == WHITE:
                    results["model_a_wins"] += 1
                else:
                    results["model_b_wins"] += 1

        return results

    # Parallel evaluation — use lightweight manual Process instead of Pool
    # to avoid "too many open files" when many workers are already running.
    model_config = _extract_model_config(model_a)
    model_a_state = {k: v.cpu() for k, v in model_a.state_dict().items()}
    model_b_state = {k: v.cpu() for k, v in model_b.state_dict().items()}

    game_module = "gomoku" if game_class.__name__ == "GomokuGame" else "othello"

    # Determine GPU allocation
    num_gpus = torch.cuda.device_count() if device.startswith("cuda") else 0

    # Cap parallelism to avoid overwhelming the system
    # When many self-play workers are already running, keep eval lean
    num_parallel = min(num_parallel, num_games, 4)

    # Build work units
    games_per_worker = num_games // num_parallel
    remainder = num_games % num_parallel
    worker_args = []
    for i in range(num_parallel):
        worker_games = games_per_worker + (1 if i < remainder else 0)
        worker_device = device
        if num_gpus > 0:
            worker_device = f"cuda:{i % num_gpus}"
        worker_args.append(
            (
                model_a_state,
                model_b_state,
                model_config,
                worker_games,
                num_simulations,
                temperature,
                worker_device,
                game_module,
            )
        )

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue(maxsize=num_parallel)
    processes = []

    for args in worker_args:
        p = ctx.Process(target=_play_match_worker_queue, args=(args, result_queue))
        p.start()
        processes.append(p)

    # Collect results
    worker_results = []
    for _ in range(num_parallel):
        worker_results.append(result_queue.get())

    # Wait for all workers to finish
    for p in processes:
        p.join(timeout=300)
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)

    # Aggregate
    results = {"model_a_wins": 0, "model_b_wins": 0, "draws": 0}
    for r in worker_results:
        for k in results:
            results[k] += r[k]

    return results


def evaluate_challenger(
    champion_model: OthelloNet,
    challenger_model: OthelloNet,
    num_games: int = 200,
    num_simulations: int = 400,
    win_threshold: float = 0.55,
    device: str = "cpu",
    game_class=None,
    num_parallel: int = 1,
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
        game_class=game_class,
        num_parallel=num_parallel,
    )

    total = results["model_a_wins"] + results["model_b_wins"] + results["draws"]
    challenger_wins = results["model_b_wins"]
    challenger_win_rate = challenger_wins / total if total > 0 else 0.0

    is_better = challenger_win_rate > win_threshold

    detailed_results = {
        "challenger_win_rate": challenger_win_rate,
        "challenger_wins": challenger_wins,
        "champion_wins": results["model_a_wins"],
        "draws": results["draws"],
        "total_games": total,
        "is_better": is_better,
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


def load_champion(checkpoint_dir: str, model: OthelloNet, device: str = "cpu") -> bool:
    """Load champion model if it exists."""
    champion_path = os.path.join(checkpoint_dir, "best_model.pt")
    if os.path.exists(champion_path):
        state_dict = torch.load(champion_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict)
        return True
    return False


def save_elo_history(elo_history: list, log_dir: str):
    """Save ELO history to JSON."""
    path = os.path.join(log_dir, "elo_history.json")
    with open(path, "w") as f:
        json.dump(elo_history, f, indent=2)
