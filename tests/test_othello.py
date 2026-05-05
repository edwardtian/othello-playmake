# Tests for Othello Game Engine

import pytest
import numpy as np
from game.othello import (
    OthelloGame, EMPTY, BLACK, WHITE, PASS_ACTION,
    BOARD_SIZE, TOTAL_ACTIONS, play_game
)


class TestOthelloInit:
    def test_initial_board(self):
        game = OthelloGame()
        assert game.board[3, 3] == WHITE
        assert game.board[3, 4] == BLACK
        assert game.board[4, 3] == BLACK
        assert game.board[4, 4] == WHITE
        assert game.current_player == BLACK

    def test_initial_legal_moves(self):
        game = OthelloGame()
        moves = game.get_legal_moves()
        # Standard Othello opening: black has 4 legal moves
        expected = [19, 26, 37, 44]  # (2,3), (3,2), (4,5), (5,4)
        assert sorted(moves) == sorted(expected)


class TestMoveValidation:
    def test_valid_move(self):
        game = OthelloGame()
        assert game.is_valid_move(19) is True  # (2,3)

    def test_invalid_move_empty(self):
        game = OthelloGame()
        assert game.is_valid_move(0) is False  # Corner is invalid

    def test_invalid_move_occupied(self):
        game = OthelloGame()
        assert game.is_valid_move(27) is False  # (3,3) is occupied

    def test_pass_invalid_when_moves_exist(self):
        game = OthelloGame()
        assert game.is_valid_move(PASS_ACTION) is False


class TestMoveExecution:
    def test_make_valid_move(self):
        game = OthelloGame()
        success, msg = game.make_move(19)  # Black plays (2,3)
        assert success is True
        assert game.board[2, 3] == BLACK
        assert game.board[3, 3] == BLACK  # Flipped
        assert game.current_player == WHITE

    def test_make_invalid_move(self):
        game = OthelloGame()
        success, msg = game.make_move(0)
        assert success is False
        # Board should remain unchanged
        assert game.board[3, 3] == WHITE

    def test_flip_multiple_stones(self):
        game = OthelloGame()
        # Set up a line: B W W W . 
        game.board[3] = [EMPTY, EMPTY, EMPTY, BLACK, WHITE, WHITE, WHITE, EMPTY]
        game.current_player = BLACK
        success, msg = game.make_move(game.coords_to_action(3, 7))
        assert success is True
        assert game.board[3, 4] == BLACK
        assert game.board[3, 5] == BLACK
        assert game.board[3, 6] == BLACK


class TestGameOver:
    def test_not_over_initially(self):
        game = OthelloGame()
        assert game.is_game_over() is False

    def test_game_over_double_pass(self):
        game = OthelloGame()
        # Force a situation where both players must pass
        # Clear the board except for a few pieces
        game.board.fill(EMPTY)
        game.board[0, 0] = BLACK
        game.board[7, 7] = WHITE
        game.current_player = BLACK
        game.pass_count = 0

        # Black has no moves, must pass
        assert PASS_ACTION in game.get_legal_moves(BLACK)
        game.make_move(PASS_ACTION, BLACK)

        # White has no moves, must pass
        assert PASS_ACTION in game.get_legal_moves(WHITE)
        game.make_move(PASS_ACTION, WHITE)

        assert game.is_game_over() is True

    def test_game_over_full_board(self):
        game = OthelloGame()
        game.board.fill(BLACK)
        assert game.is_game_over() is True


class TestWinner:
    def test_winner_black(self):
        game = OthelloGame()
        game.board.fill(EMPTY)
        game.board[0, 0] = BLACK
        game.board[0, 1] = BLACK
        game.board[0, 2] = WHITE
        game.pass_count = 2  # Force game over
        winner, b, w = game.get_winner()
        assert winner == BLACK
        assert b == 2
        assert w == 1

    def test_winner_draw(self):
        game = OthelloGame()
        game.board.fill(EMPTY)
        game.board[0, 0] = BLACK
        game.board[0, 1] = WHITE
        game.pass_count = 2
        winner, b, w = game.get_winner()
        assert winner == 0
        assert b == 1
        assert w == 1

    def test_winner_none_if_not_over(self):
        game = OthelloGame()
        winner, b, w = game.get_winner()
        assert winner is None


class TestCopy:
    def test_copy_is_independent(self):
        game = OthelloGame()
        game_copy = game.copy()
        game_copy.make_move(19)
        assert game.board[2, 3] == EMPTY
        assert game_copy.board[2, 3] == BLACK


class TestStatePlanes:
    def test_state_planes_shape(self):
        game = OthelloGame()
        planes = game.get_state_planes()
        assert planes.shape == (3, 8, 8)

    def test_state_planes_black_to_move(self):
        game = OthelloGame()
        planes = game.get_state_planes(BLACK)
        assert planes[2, 0, 0] == 1.0  # Color plane all 1s for black

    def test_state_planes_white_to_move(self):
        game = OthelloGame()
        planes = game.get_state_planes(WHITE)
        assert planes[2, 0, 0] == 0.0  # Color plane all 0s for white


class TestActionConversion:
    def test_coords_roundtrip(self):
        game = OthelloGame()
        for action in range(PASS_ACTION):
            row, col = game.action_to_coords(action)
            assert game.coords_to_action(row, col) == action


class TestPlayGame:
    def test_random_game_completes(self):
        import random

        def random_policy(game):
            moves = game.get_legal_moves()
            return random.choice(moves)

        winner, b, w, history = play_game(random_policy, random_policy)
        assert winner in [BLACK, WHITE, 0]
        assert b + w == 64 or len(history) > 0

    def test_deterministic_game(self):
        def first_move_policy(game):
            return game.get_legal_moves()[0]

        winner1, b1, w1, h1 = play_game(first_move_policy, first_move_policy)
        winner2, b2, w2, h2 = play_game(first_move_policy, first_move_policy)
        assert winner1 == winner2
        assert b1 == b2
        assert w1 == w2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
