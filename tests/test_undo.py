"""Tests for in-place board operations (make/undo roundtrip)."""

import pytest
import numpy as np
from game.othello import OthelloGame, BLACK, WHITE, EMPTY, PASS_ACTION


class TestUndoMove:
    def test_undo_single_move(self):
        """Make one move, undo it, board should be identical to start."""
        game = OthelloGame()
        initial_board = game.board.copy()
        initial_player = game.current_player

        success, msg = game.make_move(19)  # Black plays (2,3)
        assert success

        # Get the flips from history
        action, player, flips = game.move_history[-1]
        game.undo_move(action, flips, player)

        assert np.array_equal(game.board, initial_board)
        assert game.current_player == initial_player

    def test_undo_multiple_moves(self):
        """Play a sequence of moves, undo all, board should be identical to start."""
        game = OthelloGame()
        initial_board = game.board.copy()
        initial_player = game.current_player

        # Play several moves
        moves = [19, 26, 37, 44, 20, 29]
        for action in moves:
            if not game.is_valid_move(action):
                continue
            success, _ = game.make_move(action)
            assert success, f"Failed to make move {action}"

        # Undo all moves
        while game.move_history:
            success = game.pop_move()
            assert success

        assert np.array_equal(game.board, initial_board)
        assert game.current_player == initial_player

    def test_undo_pass_move(self):
        """Test undoing a pass move."""
        game = OthelloGame()
        # Clear board to force pass
        game.board.fill(EMPTY)
        game.board[0, 0] = BLACK
        game.board[7, 7] = WHITE
        game.current_player = BLACK
        game.pass_count = 0

        initial_board = game.board.copy()
        initial_player = game.current_player

        success, _ = game.make_move(PASS_ACTION)
        assert success
        assert game.pass_count == 1

        action, player, flips = game.move_history[-1]
        game.undo_move(action, flips, player)

        assert np.array_equal(game.board, initial_board)
        assert game.current_player == initial_player
        assert game.pass_count == 0

    def test_undo_complex_game(self):
        """Play a full game, undo all moves, verify initial state."""
        game = OthelloGame()
        initial_board = game.board.copy()

        # Play until game over or 60 moves
        for _ in range(60):
            if game.is_game_over():
                break
            moves = game.get_legal_moves()
            if moves:
                action = moves[0] if moves[0] != PASS_ACTION else moves[0]
                success, _ = game.make_move(action)
                assert success

        # Undo all
        while game.move_history:
            game.pop_move()

        assert np.array_equal(game.board, initial_board)
        assert game.current_player == BLACK
        assert game.pass_count == 0

    def test_undo_replay_consistency(self):
        """After undo, replaying the same moves should produce the same final state."""
        game1 = OthelloGame()
        moves = []

        # Record a sequence of moves
        for _ in range(10):
            if game1.is_game_over():
                break
            legal = game1.get_legal_moves()
            action = legal[0]
            moves.append(action)
            game1.make_move(action)

        final_board_1 = game1.board.copy()

        # Undo all
        while game1.move_history:
            game1.pop_move()

        # Replay
        for action in moves:
            game1.make_move(action)

        final_board_2 = game1.board.copy()
        assert np.array_equal(final_board_1, final_board_2)

    def test_pop_move_returns_correctly(self):
        """pop_move should return True when history exists, False when empty."""
        game = OthelloGame()
        assert game.pop_move() is False  # Empty history

        game.make_move(19)
        assert game.pop_move() is True
        assert game.pop_move() is False  # Now empty again

    def test_undo_does_not_affect_external_arrays(self):
        """Undo should only modify internal state, not external references."""
        game = OthelloGame()
        external_ref = game.board

        game.make_move(19)
        action, player, flips = game.move_history[-1]
        game.undo_move(action, flips, player)

        # external_ref should still point to the same (now restored) board
        assert np.array_equal(external_ref, game.board)

    def test_in_place_vs_copy_equivalence(self):
        """Playing with undo should match playing with copy for same moves."""
        game_copy = OthelloGame()
        game_inplace = OthelloGame()

        moves = [19, 26, 37, 44, 20]

        for action in moves:
            # Copy method
            if game_copy.is_valid_move(action):
                game_copy.make_move(action)

            # In-place method
            if game_inplace.is_valid_move(action):
                game_inplace.make_move(action)

        assert np.array_equal(game_copy.board, game_inplace.board)
        assert game_copy.current_player == game_inplace.current_player


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
