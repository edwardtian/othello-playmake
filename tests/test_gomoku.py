"""Tests for Gomoku game engine."""

import pytest
import numpy as np

from game.gomoku import GomokuGame, EMPTY, BLACK, WHITE


class TestGomokuInit:
    def test_initial_board(self):
        game = GomokuGame()
        assert game.board.shape == (15, 15)
        assert np.all(game.board == EMPTY)
        assert game.current_player == BLACK

    def test_initial_legal_moves(self):
        game = GomokuGame()
        moves = game.get_legal_moves()
        assert len(moves) == 225
        assert all(0 <= m < 225 for m in moves)


class TestMoveValidation:
    def test_valid_move(self):
        game = GomokuGame()
        assert game.is_valid_move(112)  # Center

    def test_invalid_move_occupied(self):
        game = GomokuGame()
        game.make_move(112)
        assert not game.is_valid_move(112)

    def test_invalid_move_out_of_bounds(self):
        game = GomokuGame()
        assert not game.is_valid_move(-1)
        assert not game.is_valid_move(225)


class TestMoveExecution:
    def test_make_valid_move(self):
        game = GomokuGame()
        success, msg = game.make_move(112)
        assert success
        assert game.board[7, 7] == BLACK
        assert game.current_player == WHITE

    def test_make_invalid_move(self):
        game = GomokuGame()
        game.make_move(112)
        success, msg = game.make_move(112)
        assert not success

    def test_alternating_players(self):
        game = GomokuGame()
        game.make_move(0)
        assert game.current_player == WHITE
        game.make_move(1)
        assert game.current_player == BLACK


class TestWinDetection:
    def test_horizontal_win(self):
        game = GomokuGame()
        # Black plays 5 in a row horizontally
        for col in range(5):
            game.make_move(col)  # Black
            if col < 4:
                game.make_move(col + 10)  # White plays elsewhere
        assert game.is_game_over()
        winner, bc, wc = game.get_winner()
        assert winner == BLACK

    def test_vertical_win(self):
        game = GomokuGame()
        # Black plays 5 in a row vertically
        for row in range(5):
            game.make_move(row * 15)  # Black
            if row < 4:
                game.make_move(row * 15 + 1)  # White plays elsewhere
        assert game.is_game_over()
        winner, bc, wc = game.get_winner()
        assert winner == BLACK

    def test_diagonal_win(self):
        game = GomokuGame()
        # Black plays 5 in a row diagonally
        for i in range(5):
            game.make_move(i * 15 + i)  # Black
            if i < 4:
                game.make_move(i * 15 + i + 1)  # White plays elsewhere
        assert game.is_game_over()
        winner, bc, wc = game.get_winner()
        assert winner == BLACK

    def test_anti_diagonal_win(self):
        game = GomokuGame()
        # Black plays 5 in a row anti-diagonally
        for i in range(5):
            game.make_move(i * 15 + (4 - i))  # Black
            if i < 4:
                game.make_move(i * 15 + (4 - i) + 1)  # White plays elsewhere
        assert game.is_game_over()
        winner, bc, wc = game.get_winner()
        assert winner == BLACK

    def test_win_requires_exactly_five(self):
        game = GomokuGame()
        # Place 4 in a row - not a win
        for col in range(4):
            game.make_move(col)
            game.make_move(col + 10)
        assert not game.is_game_over()


class TestGameOver:
    def test_not_over_initially(self):
        game = GomokuGame()
        assert not game.is_game_over()

    def test_game_over_full_board(self):
        game = GomokuGame()
        # Fill the board alternately (no 5-in-a-row)
        # This is hard to do without creating a win, so just check the mechanism
        # by setting board directly
        for i in range(225):
            if game.is_game_over():
                break
            moves = game.get_legal_moves()
            if moves:
                game.make_move(moves[0])
        assert game.is_game_over()

    def test_winner_none_if_not_over(self):
        game = GomokuGame()
        winner, bc, wc = game.get_winner()
        assert winner is None


class TestDraw:
    def test_draw_full_board(self):
        game = GomokuGame()
        # Fill board in a checkerboard pattern that avoids 5-in-a-row
        game.board = np.array([
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
            [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
            [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
            [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
            [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
            [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
            [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
            [2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            [1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1],
        ], dtype=np.int8)
        game.current_player = BLACK
        game.move_history = [(i, 1 if i % 2 == 0 else 2) for i in range(225)]
        assert game.is_game_over()
        winner, bc, wc = game.get_winner()
        assert winner == 0  # Draw


class TestCopy:
    def test_copy_is_independent(self):
        game = GomokuGame()
        game.make_move(112)
        game_copy = game.copy()
        game_copy.make_move(113)
        assert game.board[7, 8] == EMPTY
        assert game_copy.board[7, 8] != EMPTY


class TestStatePlanes:
    def test_state_planes_shape(self):
        game = GomokuGame()
        planes = game.get_state_planes()
        assert planes.shape == (3, 15, 15)

    def test_state_planes_black_to_move(self):
        game = GomokuGame()
        planes = game.get_state_planes()
        assert planes[2][0][0] == 1.0

    def test_state_planes_white_to_move(self):
        game = GomokuGame()
        game.make_move(112)
        planes = game.get_state_planes()
        assert planes[2][0][0] == 0.0


class TestActionConversion:
    def test_coords_roundtrip(self):
        game = GomokuGame()
        for action in [0, 112, 224]:
            row, col = game.action_to_coords(action)
            assert game.coords_to_action(row, col) == action


class TestUndoMove:
    def test_undo_single_move(self):
        game = GomokuGame()
        game.make_move(112)
        assert game.board[7, 7] == BLACK
        game.undo_move()
        assert game.board[7, 7] == EMPTY
        assert game.current_player == BLACK

    def test_undo_multiple_moves(self):
        game = GomokuGame()
        game.make_move(112)
        game.make_move(113)
        game.undo_move()
        assert game.board[7, 8] == EMPTY
        assert game.current_player == WHITE
        game.undo_move()
        assert game.board[7, 7] == EMPTY
        assert game.current_player == BLACK


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
