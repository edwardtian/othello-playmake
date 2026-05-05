"""Tests for Othello Neural Network."""

import pytest
import torch
import numpy as np
from ai.model import OthelloNet, create_model
from game.othello import OthelloGame


class TestModelArchitecture:
    def test_model_creation(self):
        model = OthelloNet(num_blocks=2, num_channels=64)
        assert model is not None

    def test_forward_pass_shape(self):
        model = OthelloNet(num_blocks=2, num_channels=64)
        batch_size = 4
        x = torch.randn(batch_size, 3, 8, 8)
        policy_logits, value = model(x)
        assert policy_logits.shape == (batch_size, 65)
        assert value.shape == (batch_size, 1)

    def test_single_input(self):
        model = OthelloNet(num_blocks=2, num_channels=64)
        x = torch.randn(1, 3, 8, 8)
        policy_logits, value = model(x)
        assert policy_logits.shape == (1, 65)
        assert value.shape == (1, 1)

    def test_value_range(self):
        model = OthelloNet(num_blocks=2, num_channels=64)
        x = torch.randn(10, 3, 8, 8)
        _, value = model(x)
        assert torch.all(value >= -1.0)
        assert torch.all(value <= 1.0)

    def test_predict_method(self):
        model = OthelloNet(num_blocks=2, num_channels=64)
        # Test with batch
        x = torch.randn(4, 3, 8, 8)
        policy, value = model.predict(x)
        assert policy.shape == (4, 65)
        assert value.shape == (4, 1)
        assert torch.allclose(policy.sum(dim=1), torch.ones(4), atol=1e-5)

    def test_predict_single(self):
        model = OthelloNet(num_blocks=2, num_channels=64)
        x = torch.randn(3, 8, 8)
        policy, value = model.predict(x)
        assert policy.shape == (1, 65)
        assert value.shape == (1, 1)

    def test_parameter_count(self):
        model = OthelloNet(num_blocks=2, num_channels=64)
        count = model.count_parameters()
        assert count > 0
        # Small model should have < 1M params
        assert count < 1_000_000


class TestModelWithGameState:
    def test_model_on_initial_state(self):
        model = OthelloNet(num_blocks=2, num_channels=64)
        game = OthelloGame()
        state = game.get_state_planes()
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        policy_logits, value = model(state_tensor)
        assert policy_logits.shape == (1, 65)
        assert value.shape == (1, 1)

    def test_model_on_random_state(self):
        model = OthelloNet(num_blocks=2, num_channels=64)
        for _ in range(10):
            game = OthelloGame()
            # Make a few random moves
            for _ in range(5):
                moves = game.get_legal_moves()
                if moves:
                    import random
                    action = random.choice(moves)
                    game.make_move(action)
            state = game.get_state_planes()
            state_tensor = torch.from_numpy(state).float().unsqueeze(0)
            policy_logits, value = model(state_tensor)
            assert policy_logits.shape == (1, 65)
            assert value.shape == (1, 1)


class TestCreateModel:
    def test_create_model_cpu(self):
        model = create_model(num_blocks=2, num_channels=64, device='cpu')
        assert next(model.parameters()).device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_create_model_cuda(self):
        model = create_model(num_blocks=2, num_channels=64, device='cuda')
        assert next(model.parameters()).device.type == 'cuda'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
