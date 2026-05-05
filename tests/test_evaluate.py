"""Tests for evaluation arena."""

import pytest
import numpy as np
import os
import tempfile
import torch
from ai.model import OthelloNet
from ai.evaluate import (
    play_match,
    evaluate_challenger,
    update_elo,
    load_champion,
    save_elo_history,
)


class TestPlayMatch:
    def test_play_match_runs(self):
        model_a = OthelloNet(num_blocks=2, num_channels=64)
        model_b = OthelloNet(num_blocks=2, num_channels=64)
        results = play_match(model_a, model_b, num_games=4, num_simulations=10, device='cpu')

        assert 'model_a_wins' in results
        assert 'model_b_wins' in results
        assert 'draws' in results
        total = results['model_a_wins'] + results['model_b_wins'] + results['draws']
        assert total == 4

    def test_identical_models_draw_or_split(self):
        model = OthelloNet(num_blocks=2, num_channels=64)
        # Clone the model
        model_b = OthelloNet(num_blocks=2, num_channels=64)
        model_b.load_state_dict(model.state_dict())

        results = play_match(model, model_b, num_games=10, num_simulations=10, device='cpu')
        total = results['model_a_wins'] + results['model_b_wins'] + results['draws']
        assert total == 10
        # With identical models and alternating colors, wins should be roughly balanced


class TestEvaluateChallenger:
    def test_evaluate_runs(self):
        champion = OthelloNet(num_blocks=2, num_channels=64)
        challenger = OthelloNet(num_blocks=2, num_channels=64)
        # Make challenger different
        for p in challenger.parameters():
            p.data += torch.randn_like(p) * 0.1

        is_better, results = evaluate_challenger(
            champion, challenger, num_games=10, num_simulations=10, device='cpu'
        )

        assert isinstance(is_better, bool)
        assert 'challenger_win_rate' in results
        assert 0.0 <= results['challenger_win_rate'] <= 1.0

    def test_evaluate_with_few_games(self):
        champion = OthelloNet(num_blocks=2, num_channels=64)
        challenger = OthelloNet(num_blocks=2, num_channels=64)
        # Make challenger different
        for p in challenger.parameters():
            p.data += torch.randn_like(p) * 0.1

        is_better, results = evaluate_challenger(
            champion, challenger, num_games=10, num_simulations=10, win_threshold=0.55, device='cpu'
        )
        # Just verify it runs and returns valid results
        assert isinstance(is_better, bool)
        assert 0.0 <= results['challenger_win_rate'] <= 1.0
        assert results['total_games'] == 10


class TestELO:
    def test_elo_update_win(self):
        champ, chall = update_elo(1500, 1500, 1.0)
        assert chall > 1500
        assert champ < 1500

    def test_elo_update_loss(self):
        champ, chall = update_elo(1500, 1500, 0.0)
        assert chall < 1500
        assert champ > 1500

    def test_elo_update_draw(self):
        champ, chall = update_elo(1500, 1500, 0.5)
        assert abs(champ - 1500) < 1
        assert abs(chall - 1500) < 1

    def test_elo_stronger_player_gains_less(self):
        # Strong champion beats weak challenger
        champ, chall = update_elo(1800, 1500, 0.0)  # Challenger loses
        # Champion should gain very little
        assert champ > 1800
        assert champ < 1805


class TestSaveLoad:
    def test_save_load_elo(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history = [{'step': 100, 'champion_elo': 1500, 'challenger_elo': 1500}]
            save_elo_history(history, tmpdir)
            assert os.path.exists(os.path.join(tmpdir, 'elo_history.json'))

    def test_load_champion_missing(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = OthelloNet(num_blocks=2, num_channels=64)
            success = load_champion(tmpdir, model, 'cpu')
            assert success is False

    def test_load_champion_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            model = OthelloNet(num_blocks=2, num_channels=64)
            torch.save(model.state_dict(), os.path.join(tmpdir, 'best_model.pt'))
            model2 = OthelloNet(num_blocks=2, num_channels=64)
            success = load_champion(tmpdir, model2, 'cpu')
            assert success is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
