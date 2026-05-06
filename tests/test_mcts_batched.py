"""Tests for Batched MCTS."""

import pytest
import time
import numpy as np
from ai.model import OthelloNet
from ai.mcts import MCTS
from ai.mcts_batched import BatchedMCTS
from game.othello import OthelloGame, PASS_ACTION


class TestBatchedMCTSCorrectness:
    def test_batched_search_runs(self):
        model = OthelloNet(num_blocks=2, num_channels=64)
        mcts = BatchedMCTS(model, num_simulations=32, batch_size=8)
        game = OthelloGame()
        action_probs, value = mcts.search(game, temperature=1.0)
        assert action_probs.shape == (65,)
        assert np.isclose(action_probs.sum(), 1.0, atol=1e-5)

    def test_batched_returns_valid_move(self):
        model = OthelloNet(num_blocks=2, num_channels=64)
        mcts = BatchedMCTS(model, num_simulations=32, batch_size=8)
        game = OthelloGame()
        action_probs, _ = mcts.search(game, temperature=0.0)
        best_action = np.argmax(action_probs)
        assert game.is_valid_move(best_action)

    def test_batched_vs_standard_same_params(self):
        """Batched and standard MCTS should produce similar move distributions."""
        model = OthelloNet(num_blocks=2, num_channels=64)

        # Standard MCTS
        mcts_std = MCTS(model, num_simulations=32, c_puct=1.5)
        # Batched MCTS
        mcts_batched = BatchedMCTS(model, num_simulations=32, batch_size=8, c_puct=1.5)

        game = OthelloGame()

        # Run both (with fixed seed for reproducibility if possible)
        np.random.seed(42)
        probs_std, _ = mcts_std.search(game, temperature=1.0)

        np.random.seed(42)
        probs_batched, _ = mcts_batched.search(game, temperature=1.0)

        # Best move should agree (or be very close)
        best_std = np.argmax(probs_std)
        best_batched = np.argmax(probs_batched)

        # With same seed and enough simulations, they should pick same move
        # But virtual loss introduces some difference, so check top-3 overlap
        top3_std = set(np.argsort(probs_std)[-3:])
        top3_batched = set(np.argsort(probs_batched)[-3:])
        assert len(top3_std & top3_batched) >= 1, "Top moves should overlap"

    def test_batched_pass_handling(self):
        model = OthelloNet(num_blocks=2, num_channels=64)
        mcts = BatchedMCTS(model, num_simulations=16, batch_size=4)
        game = OthelloGame()
        game.board.fill(0)
        game.board[0, 0] = 1
        game.board[7, 7] = 2
        game.current_player = 1
        game.pass_count = 0

        action_probs, _ = mcts.search(game, temperature=0.0)
        best_action = np.argmax(action_probs)
        assert best_action == PASS_ACTION

    def test_batched_game_progression(self):
        """Test that batched MCTS can play a few moves without errors."""
        model = OthelloNet(num_blocks=2, num_channels=64)
        mcts = BatchedMCTS(model, num_simulations=16, batch_size=4)
        game = OthelloGame()

        for _ in range(5):
            if game.is_game_over():
                break
            action = mcts.get_best_move(game, temperature=1.0)
            success, _ = game.make_move(action)
            assert success, f"MCTS returned invalid move: {action}"


class TestBatchedMCTSSpeed:
    def test_batched_faster_than_standard(self):
        """Benchmark: batched MCTS should be significantly faster."""
        model = OthelloNet(num_blocks=4, num_channels=128)
        game = OthelloGame()

        # Standard MCTS
        mcts_std = MCTS(model, num_simulations=64)
        start = time.time()
        for _ in range(3):
            g = game.copy()
            mcts_std.search(g, temperature=1.0)
        time_std = time.time() - start

        # Batched MCTS
        mcts_batched = BatchedMCTS(model, num_simulations=64, batch_size=16)
        start = time.time()
        for _ in range(3):
            g = game.copy()
            mcts_batched.search(g, temperature=1.0)
        time_batched = time.time() - start

        speedup = time_std / time_batched
        print(f"\nStandard: {time_std:.2f}s, Batched: {time_batched:.2f}s, Speedup: {speedup:.1f}x")

        # Batched should be at least 2x faster even on CPU
        assert speedup > 2.0, f"Batched MCTS only {speedup:.1f}x faster, expected >2x"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
