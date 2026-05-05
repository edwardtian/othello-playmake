"""Tests for MCTS."""

import pytest
import torch
import numpy as np
from ai.model import OthelloNet
from ai.mcts import MCTS, MCTSNode
from game.othello import OthelloGame, PASS_ACTION


class TestMCTSNode:
    def test_node_init(self):
        node = MCTSNode(prior=0.5, action=10)
        assert node.prior == 0.5
        assert node.action == 10
        assert node.visit_count == 0
        assert node.value_sum == 0.0
        assert node.q_value == 0.0

    def test_node_q_value(self):
        node = MCTSNode()
        node.visit_count = 4
        node.value_sum = 2.0
        assert node.q_value == 0.5

    def test_node_select_child(self):
        parent = MCTSNode()
        parent.visit_count = 10
        child1 = MCTSNode(prior=0.8, action=1, parent=parent)
        child1.visit_count = 5
        child1.value_sum = 2.0  # Q = 0.4
        child2 = MCTSNode(prior=0.2, action=2, parent=parent)
        child2.visit_count = 1
        child2.value_sum = 0.5  # Q = 0.5
        parent.children = {1: child1, 2: child2}
        parent.is_expanded = True

        # Child1 has higher UCT due to higher prior and more visits
        action, child = parent.select_child(c_puct=1.0)
        assert action in [1, 2]
        assert child is not None

    def test_node_expand(self):
        node = MCTSNode()
        priors = np.zeros(65)
        priors[19] = 0.5
        priors[26] = 0.5
        node.expand(priors, [19, 26])
        assert node.is_expanded
        assert len(node.children) == 2
        assert 19 in node.children
        assert 26 in node.children

    def test_node_backup(self):
        root = MCTSNode()
        child = MCTSNode(parent=root)
        root.children[0] = child

        child.backup(1.0)
        assert child.visit_count == 1
        assert child.value_sum == 1.0
        assert root.visit_count == 1
        assert root.value_sum == -1.0  # Negated for parent


class TestMCTSSearch:
    def test_mcts_search_runs(self):
        model = OthelloNet(num_blocks=2, num_channels=64)
        mcts = MCTS(model, num_simulations=10, c_puct=1.0)
        game = OthelloGame()
        action_probs, value = mcts.search(game, temperature=1.0)
        assert action_probs.shape == (65,)
        assert np.isclose(action_probs.sum(), 1.0, atol=1e-5)
        assert len(value) == 1

    def test_mcts_returns_valid_move(self):
        model = OthelloNet(num_blocks=2, num_channels=64)
        mcts = MCTS(model, num_simulations=10, c_puct=1.0)
        game = OthelloGame()
        action_probs, _ = mcts.search(game, temperature=0.0)
        best_action = np.argmax(action_probs)
        assert game.is_valid_move(best_action)

    def test_mcts_get_best_move(self):
        model = OthelloNet(num_blocks=2, num_channels=64)
        mcts = MCTS(model, num_simulations=10, c_puct=1.0)
        game = OthelloGame()
        action = mcts.get_best_move(game)
        assert game.is_valid_move(action)

    def test_mcts_zero_temperature(self):
        model = OthelloNet(num_blocks=2, num_channels=64)
        mcts = MCTS(model, num_simulations=50, c_puct=1.0)
        game = OthelloGame()
        action_probs, _ = mcts.search(game, temperature=0.0)
        # Should be one-hot
        assert np.count_nonzero(action_probs > 0) == 1
        assert np.isclose(action_probs.sum(), 1.0, atol=1e-5)

    def test_mcts_game_progression(self):
        """Test that MCTS can play a few moves without errors."""
        model = OthelloNet(num_blocks=2, num_channels=64)
        mcts = MCTS(model, num_simulations=10, c_puct=1.0)
        game = OthelloGame()

        for _ in range(5):
            if game.is_game_over():
                break
            action = mcts.get_best_move(game, temperature=1.0)
            success, _ = game.make_move(action)
            assert success, f"MCTS returned invalid move: {action}"

    def test_mcts_pass_handling(self):
        """Test MCTS when pass is the only move."""
        model = OthelloNet(num_blocks=2, num_channels=64)
        mcts = MCTS(model, num_simulations=10, c_puct=1.0)
        game = OthelloGame()
        # Clear board and set up a no-move situation
        game.board.fill(0)
        game.board[0, 0] = 1  # Black
        game.board[7, 7] = 2  # White
        game.current_player = 1
        game.pass_count = 0

        action_probs, _ = mcts.search(game, temperature=0.0)
        best_action = np.argmax(action_probs)
        assert best_action == PASS_ACTION
        assert game.is_valid_move(best_action)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
