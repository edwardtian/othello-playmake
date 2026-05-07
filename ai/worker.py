"""
Self-Play Worker for Othello AI.

Runs in a dedicated process, plays games using BatchedMCTS with
external inference server for neural network evaluation.
"""

import time
import numpy as np
import torch
import torch.multiprocessing as mp
from typing import List, Tuple, Dict, Any
from game.othello import OthelloGame
from ai.mcts_batched import BatchedMCTS


class InferenceClient:
    """
    Client that connects to InferenceServer via queues.
    Each client has its own dedicated result queue to prevent cross-contamination.
    """

    def __init__(
        self,
        request_queue: mp.Queue,
        result_queue: mp.Queue,
        worker_id: int,
        local_batch_size: int = 8,
        timeout: float = 10.0,
        action_size: int = 65,
    ):
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.worker_id = worker_id
        self.local_batch_size = local_batch_size
        self.timeout = timeout
        self.action_size = action_size
        self.request_counter = 0

    def evaluate(self, state: np.ndarray) -> Tuple[np.ndarray, float]:
        """Evaluate a single state."""
        req_id = f"w{self.worker_id}_r{self.request_counter}"
        self.request_counter += 1

        self.request_queue.put({
            'request_id': req_id,
            'state': state,
        }, block=True, timeout=self.timeout)

        # Wait for result from dedicated queue
        deadline = time.time() + self.timeout
        while time.time() < deadline:
            try:
                result = self.result_queue.get(timeout=0.01)
                if result['request_id'] == req_id:
                    return result['policy'], result['value']
            except:
                continue

        # Timeout fallback
        print(f"[Worker {self.worker_id}] Inference timeout, using uniform")
        return np.ones(self.action_size, dtype=np.float32) / self.action_size, 0.0

    def evaluate_batch(self, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Evaluate a batch of states.
        
        Args:
            states: (B, 3, 8, 8) numpy array
            
        Returns:
            policies: (B, 65) numpy array
            values: (B,) numpy array
        """
        batch_size = states.shape[0]
        policies = np.zeros((batch_size, self.action_size), dtype=np.float32)
        values = np.zeros(batch_size, dtype=np.float32)

        # Send all requests
        request_ids = []
        for i in range(batch_size):
            req_id = f"w{self.worker_id}_r{self.request_counter}"
            self.request_counter += 1
            request_ids.append(req_id)
            self.request_queue.put({
                'request_id': req_id,
                'state': states[i],
            }, block=True, timeout=self.timeout)

        # Collect all results from dedicated queue
        collected = 0
        deadline = time.time() + self.timeout
        while collected < batch_size and time.time() < deadline:
            try:
                result = self.result_queue.get(timeout=0.01)
                req_id = result['request_id']
                if req_id in request_ids:
                    idx = request_ids.index(req_id)
                    policies[idx] = result['policy']
                    values[idx] = result['value']
                    collected += 1
            except:
                continue

        # Fill any missing with uniform
        for i in range(batch_size):
            req_id = request_ids[i]
            if policies[i].sum() < 0.99:  # Not filled
                policies[i] = np.ones(self.action_size, dtype=np.float32) / self.action_size
                values[i] = 0.0

        return policies, values


class SelfPlayWorker:
    """
    Worker process that generates self-play games.
    """

    def __init__(
        self,
        worker_id: int,
        request_queue: mp.Queue,
        result_queue: mp.Queue,
        game_queue: mp.Queue,
        mcts_config: Dict[str, Any],
        temperature_schedule: List[Tuple[int, float]] = None,
        action_size: int = 65,
    ):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.result_queue = result_queue
        self.game_queue = game_queue
        self.mcts_config = mcts_config
        self.temperature_schedule = temperature_schedule or [(0, 1.0)]
        self.action_size = action_size

    def _get_temperature(self, move_count: int) -> float:
        """Get temperature for current move count."""
        for threshold, temp in reversed(self.temperature_schedule):
            if move_count >= threshold:
                return temp
        return 1.0

    def run(self, num_games: int = None):
        """
        Main worker loop.
        
        Args:
            num_games: Number of games to play (None = infinite)
        """
        client = InferenceClient(
            self.request_queue,
            self.result_queue,
            self.worker_id,
            action_size=self.action_size,
        )

        # Create MCTS with external evaluator
        # Remove 'evaluator' from config if present to avoid conflict
        mcts_cfg = dict(self.mcts_config)
        mcts_cfg.pop('evaluator', None)
        mcts_cfg.pop('model', None)
        
        mcts = BatchedMCTS(
            evaluator=client.evaluate_batch,
            **mcts_cfg,
        )

        games_played = 0
        while num_games is None or games_played < num_games:
            try:
                game_data = self._play_one_game(mcts)
                self.game_queue.put(game_data, block=True, timeout=30)
                games_played += 1

                if games_played % 10 == 0:
                    print(f"[Worker {self.worker_id}] Played {games_played} games")

            except Exception as e:
                print(f"[Worker {self.worker_id}] Error: {e}")
                time.sleep(1)

    def _play_one_game(self, mcts: BatchedMCTS) -> List[Tuple]:
        """Play one complete self-play game."""
        game = OthelloGame()
        game_history = []
        move_count = 0

        while not game.is_game_over():
            temperature = self._get_temperature(move_count)
            action_probs, _ = mcts.search(game, temperature=temperature)
            action = np.random.choice(len(action_probs), p=action_probs)

            state = game.get_state_planes()
            game_history.append((state, action_probs.copy(), action, game.current_player))

            success, _ = game.make_move(action)
            if not success:
                raise RuntimeError(f"Invalid move: {action}")

            move_count += 1

        # Determine winner and assign outcomes
        winner, _, _ = game.get_winner()
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


def start_worker_pool(
    num_workers: int,
    request_queue: mp.Queue,
    result_queues: List[mp.Queue],
    game_queue: mp.Queue,
    mcts_config: Dict[str, Any],
    action_size: int = 65,
) -> List[mp.Process]:
    """
    Start a pool of self-play workers.
    
    Args:
        num_workers: Number of workers to start
        request_queue: Shared queue for sending evaluation requests to inference server
        result_queues: List of queues, one per worker, for receiving results
        game_queue: Queue for sending completed games back to main process
        mcts_config: MCTS configuration dict
        action_size: Number of possible actions
    
    Returns:
        List of worker processes
    """
    ctx = mp.get_context('spawn')
    processes = []

    for i in range(num_workers):
        worker = SelfPlayWorker(
            worker_id=i,
            request_queue=request_queue,
            result_queue=result_queues[i],
            game_queue=game_queue,
            mcts_config=mcts_config,
            action_size=action_size,
        )
        p = ctx.Process(
            target=worker.run,
            daemon=True,
        )
        p.start()
        processes.append(p)
        print(f"[Main] Started worker {i}/{num_workers}")

    return processes
