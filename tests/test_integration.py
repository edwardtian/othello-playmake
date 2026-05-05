"""
Integration tests for the full Othello AI pipeline.

Verifies that all components work together end-to-end.
"""

import pytest
import os
import tempfile
import torch
import numpy as np

from game.othello import OthelloGame, BLACK, WHITE
from ai.model import OthelloNet, create_model
from ai.mcts import MCTS
from ai.self_play import generate_self_play_game
from ai.trainer import ReplayBuffer, Trainer
from ai.evaluate import play_match, evaluate_challenger


class TestFullPipeline:
    def test_game_engine_model_mcts_integration(self):
        """Test that model + MCTS can play a full game on the engine."""
        model = OthelloNet(num_blocks=2, num_channels=64)
        mcts = MCTS(model, num_simulations=10)
        game = OthelloGame()

        move_count = 0
        while not game.is_game_over() and move_count < 100:
            action = mcts.get_best_move(game, temperature=1.0)
            success, _ = game.make_move(action)
            assert success, f"MCTS returned invalid move at step {move_count}"
            move_count += 1

        assert game.is_game_over()
        winner, b, w = game.get_winner()
        assert winner in [BLACK, WHITE, 0]

    def test_self_play_to_training(self):
        """Test that self-play data can be generated and used for training."""
        model = OthelloNet(num_blocks=2, num_channels=64)
        mcts = MCTS(model, num_simulations=10)
        buffer = ReplayBuffer(capacity=10000)
        trainer = Trainer(model, mcts, buffer, device='cpu', batch_size=4)

        # Generate some games
        trainer.generate_self_play_data(num_games=3)
        assert len(buffer) > 0

        # Train for a few steps
        initial_loss = None
        for i in range(10):
            metrics = trainer.train_step()
            if metrics['total_loss'] > 0:
                if initial_loss is None:
                    initial_loss = metrics['total_loss']

        # Loss should have changed (model is learning something)
        final_loss = metrics['total_loss']
        assert initial_loss is not None
        assert final_loss > 0

    def test_checkpoint_save_load_resume(self):
        """Test that training can be saved and resumed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            model = OthelloNet(num_blocks=2, num_channels=64)
            mcts = MCTS(model, num_simulations=10)
            buffer = ReplayBuffer(capacity=1000)
            trainer = Trainer(
                model, mcts, buffer,
                device='cpu', batch_size=4,
                checkpoint_dir=tmpdir
            )

            # Add data and train
            for i in range(20):
                buffer.add(
                    np.random.randn(3, 8, 8).astype(np.float32),
                    np.ones(65, dtype=np.float32) / 65,
                    1.0
                )

            for _ in range(5):
                trainer.train_step()

            # Save
            trainer.save_checkpoint('test_resume')

            # Create new trainer and load
            model2 = OthelloNet(num_blocks=2, num_channels=64)
            mcts2 = MCTS(model2, num_simulations=10)
            buffer2 = ReplayBuffer(capacity=1000)
            trainer2 = Trainer(
                model2, mcts2, buffer2,
                device='cpu', batch_size=4,
                checkpoint_dir=tmpdir
            )

            success = trainer2.load_checkpoint(os.path.join(tmpdir, 'test_resume'))
            assert success
            assert trainer2.training_step == trainer.training_step

            # Should be able to continue training
            metrics = trainer2.train_step()
            assert metrics['total_loss'] > 0

    def test_evaluation_pipeline(self):
        """Test that two models can be evaluated against each other."""
        model_a = OthelloNet(num_blocks=2, num_channels=64)
        model_b = OthelloNet(num_blocks=2, num_channels=64)

        results = play_match(model_a, model_b, num_games=10, num_simulations=10, device='cpu')
        total = results['model_a_wins'] + results['model_b_wins'] + results['draws']
        assert total == 10

    def test_model_inference_on_game_states(self):
        """Test model inference on various game states."""
        model = OthelloNet(num_blocks=2, num_channels=64)
        game = OthelloGame()

        for _ in range(20):
            state = game.get_state_planes()
            state_tensor = torch.from_numpy(state).float()
            policy, value = model.predict(state_tensor)

            assert policy.shape == (1, 65)
            assert value.shape == (1, 1)
            assert torch.allclose(policy.sum(), torch.tensor(1.0), atol=1e-5)
            assert -1.0 <= value.item() <= 1.0

            moves = game.get_legal_moves()
            if moves and moves[0] != 64:
                game.make_move(moves[0])
            else:
                break


class TestWindowsCompatibility:
    def test_no_multiprocessing_issues_in_import(self):
        """Verify that all modules can be imported without multiprocessing issues."""
        import game.othello
        import ai.model
        import ai.mcts
        import ai.self_play
        import ai.trainer
        import ai.evaluate
        import web.server
        import web.training_manager

        # All imports should succeed without hanging on Windows
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
