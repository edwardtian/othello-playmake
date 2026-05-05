"""Tests for Trainer and ReplayBuffer."""

import pytest
import numpy as np
import torch
import os
import tempfile
from ai.model import OthelloNet
from ai.mcts import MCTS
from ai.trainer import ReplayBuffer, Trainer


class TestReplayBuffer:
    def test_buffer_init(self):
        buf = ReplayBuffer(capacity=100)
        assert len(buf) == 0

    def test_buffer_add(self):
        buf = ReplayBuffer(capacity=100)
        state = np.zeros((3, 8, 8), dtype=np.float32)
        policy = np.ones(65, dtype=np.float32) / 65
        buf.add(state, policy, 1.0)
        assert len(buf) == 1

    def test_buffer_capacity(self):
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            state = np.zeros((3, 8, 8), dtype=np.float32)
            policy = np.ones(65, dtype=np.float32) / 65
            buf.add(state, policy, float(i))
        assert len(buf) == 5

    def test_buffer_sample(self):
        buf = ReplayBuffer(capacity=100)
        for i in range(50):
            state = np.zeros((3, 8, 8), dtype=np.float32)
            policy = np.ones(65, dtype=np.float32) / 65
            buf.add(state, policy, float(i))

        states, policies, values = buf.sample(10, device='cpu')
        assert states.shape == (10, 3, 8, 8)
        assert policies.shape == (10, 65)
        assert values.shape == (10, 1)

    def test_buffer_sample_small(self):
        buf = ReplayBuffer(capacity=100)
        buf.add(np.zeros((3, 8, 8)), np.ones(65) / 65, 1.0)

        states, policies, values = buf.sample(10, device='cpu')
        assert states.shape[0] == 1  # Should adjust batch size

    def test_buffer_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            buf = ReplayBuffer(capacity=100)
            for i in range(10):
                state = np.ones((3, 8, 8), dtype=np.float32) * i
                policy = np.ones(65, dtype=np.float32) / 65
                buf.add(state, policy, float(i))

            path = os.path.join(tmpdir, 'buffer.pt')
            buf.save(path)

            buf2 = ReplayBuffer(capacity=100)
            buf2.load(path)
            assert len(buf2) == 10

    def test_buffer_clear(self):
        buf = ReplayBuffer(capacity=100)
        buf.add(np.zeros((3, 8, 8)), np.ones(65) / 65, 1.0)
        buf.clear()
        assert len(buf) == 0


class TestTrainer:
    def test_trainer_init(self):
        model = OthelloNet(num_blocks=2, num_channels=64)
        mcts = MCTS(model, num_simulations=10)
        buffer = ReplayBuffer(capacity=1000)
        trainer = Trainer(model, mcts, buffer, device='cpu', batch_size=4)
        assert trainer.training_step == 0

    def test_train_step_empty_buffer(self):
        model = OthelloNet(num_blocks=2, num_channels=64)
        mcts = MCTS(model, num_simulations=10)
        buffer = ReplayBuffer(capacity=1000)
        trainer = Trainer(model, mcts, buffer, device='cpu', batch_size=4)

        metrics = trainer.train_step()
        assert metrics['total_loss'] == 0.0

    def test_train_step_with_data(self):
        model = OthelloNet(num_blocks=2, num_channels=64)
        mcts = MCTS(model, num_simulations=10)
        buffer = ReplayBuffer(capacity=1000)

        # Add synthetic data
        for i in range(20):
            state = np.random.randn(3, 8, 8).astype(np.float32)
            policy = np.ones(65, dtype=np.float32) / 65
            buffer.add(state, policy, 1.0)

        trainer = Trainer(model, mcts, buffer, device='cpu', batch_size=4)
        metrics = trainer.train_step()
        assert metrics['total_loss'] > 0
        assert metrics['policy_loss'] > 0
        assert metrics['value_loss'] > 0

    def test_checkpoint_save_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = OthelloNet(num_blocks=2, num_channels=64)
            mcts = MCTS(model, num_simulations=10)
            buffer = ReplayBuffer(capacity=1000)

            # Add some data
            for i in range(10):
                buffer.add(np.ones((3, 8, 8)) * i, np.ones(65) / 65, float(i))

            trainer = Trainer(
                model, mcts, buffer,
                device='cpu', batch_size=4,
                checkpoint_dir=tmpdir
            )
            trainer.training_step = 42
            trainer.games_played = 100

            # Save
            trainer.save_checkpoint('test_ckpt')

            # Load
            model2 = OthelloNet(num_blocks=2, num_channels=64)
            mcts2 = MCTS(model2, num_simulations=10)
            buffer2 = ReplayBuffer(capacity=1000)
            trainer2 = Trainer(
                model2, mcts2, buffer2,
                device='cpu', batch_size=4,
                checkpoint_dir=tmpdir
            )
            success = trainer2.load_checkpoint(os.path.join(tmpdir, 'test_ckpt'))
            assert success
            assert trainer2.training_step == 42
            assert trainer2.games_played == 100
            assert len(trainer2.replay_buffer) == 10

    def test_generate_self_play_data(self):
        model = OthelloNet(num_blocks=2, num_channels=64)
        mcts = MCTS(model, num_simulations=10)
        buffer = ReplayBuffer(capacity=10000)
        trainer = Trainer(model, mcts, buffer, device='cpu', batch_size=4)

        positions = trainer.generate_self_play_data(num_games=2)
        assert positions > 0
        assert len(buffer) == positions
        assert trainer.games_played == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
