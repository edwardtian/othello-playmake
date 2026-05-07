"""
Gomoku (Five-in-a-Row) Game Engine.

Rules:
  - 15x15 board
  - Two players: Black and White
  - Black plays first
  - On each turn, place a stone on an empty intersection
  - First player to get 5 consecutive stones (horizontal, vertical, or diagonal) wins
  - If board is full with no winner, it's a draw
"""

import numpy as np
from typing import List, Tuple, Optional

# Board values
EMPTY = 0
BLACK = 1
WHITE = 2

BOARD_SIZE = 15
TOTAL_ACTIONS = BOARD_SIZE * BOARD_SIZE

# Direction vectors for checking lines
directions = [
    (0, 1),   # horizontal
    (1, 0),   # vertical
    (1, 1),   # diagonal \
    (1, -1),  # diagonal /
]


class GomokuGame:
    """
    Gomoku game state manager.
    """

    def __init__(self, board_size: int = BOARD_SIZE):
        self.board_size = board_size
        self.total_actions = board_size * board_size
        self.board = np.zeros((board_size, board_size), dtype=np.int8)
        self.current_player = BLACK
        self.move_history = []
        self.winner = None

    def reset(self):
        """Reset to initial state."""
        self.board.fill(EMPTY)
        self.current_player = BLACK
        self.move_history.clear()
        self.winner = None
        return self

    def copy(self) -> 'GomokuGame':
        """Create a deep copy of the game state."""
        new_game = GomokuGame.__new__(GomokuGame)
        new_game.board_size = self.board_size
        new_game.total_actions = self.total_actions
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.move_history = self.move_history.copy()
        new_game.winner = self.winner
        return new_game

    def get_opponent(self, player: int) -> int:
        """Return the opponent of the given player."""
        return WHITE if player == BLACK else BLACK

    def _in_bounds(self, row: int, col: int) -> bool:
        """Check if coordinates are within the board."""
        return 0 <= row < self.board_size and 0 <= col < self.board_size

    def _check_win(self, row: int, col: int, player: int) -> bool:
        """Check if placing a stone at (row, col) creates 5 in a row for player."""
        for dr, dc in directions:
            count = 1  # The stone just placed

            # Check positive direction
            r, c = row + dr, col + dc
            while self._in_bounds(r, c) and self.board[r, c] == player:
                count += 1
                r += dr
                c += dc

            # Check negative direction
            r, c = row - dr, col - dc
            while self._in_bounds(r, c) and self.board[r, c] == player:
                count += 1
                r -= dr
                c -= dc

            if count >= 5:
                return True

        return False

    def is_valid_move(self, action: int, player: Optional[int] = None) -> bool:
        """Check if an action is valid."""
        if player is None:
            player = self.current_player

        if action < 0 or action >= self.total_actions:
            return False

        row, col = action // self.board_size, action % self.board_size
        return self.board[row, col] == EMPTY

    def get_legal_moves(self, player: Optional[int] = None) -> List[int]:
        """Return a list of all legal action indices."""
        if self.winner is not None:
            return []

        moves = []
        for action in range(self.total_actions):
            row, col = action // self.board_size, action % self.board_size
            if self.board[row, col] == EMPTY:
                moves.append(action)

        return moves

    def make_move(self, action: int, player: Optional[int] = None) -> Tuple[bool, str]:
        """Execute a move. Returns (success, message)."""
        if player is None:
            player = self.current_player

        if not self.is_valid_move(action, player):
            return False, f"Invalid move: action={action} for player={player}"

        row, col = action // self.board_size, action % self.board_size
        self.board[row, col] = player
        self.move_history.append((action, player))

        # Check for win
        if self._check_win(row, col, player):
            self.winner = player
            return True, f"Move ({row},{col}) by player {player} - WINNER!"

        # Check for draw (board full)
        if len(self.move_history) >= self.total_actions:
            self.winner = 0  # Draw
            return True, f"Move ({row},{col}) by player {player} - Board full, draw!"

        # Switch player
        self.current_player = self.get_opponent(player)

        return True, f"Move ({row},{col}) by player {player}"

    def undo_move(self):
        """Undo the last move."""
        if not self.move_history:
            return False

        action, player = self.move_history.pop()
        row, col = action // self.board_size, action % self.board_size
        self.board[row, col] = EMPTY
        self.current_player = player
        self.winner = None
        return True

    def is_game_over(self) -> bool:
        """Game is over if someone won or board is full."""
        if self.winner is not None:
            return True
        # Check if board is full
        if len(self.move_history) >= self.total_actions:
            return True
        return False

    def get_winner(self) -> Tuple[Optional[int], int, int]:
        """
        Returns (winner, black_count, white_count).
        winner: 1=BLACK wins, 2=WHITE wins, 0=DRAW, None=game not over
        """
        black_count = np.count_nonzero(self.board == BLACK)
        white_count = np.count_nonzero(self.board == WHITE)

        if self.winner is not None:
            return self.winner, black_count, white_count

        # If board is full with no winner, it's a draw
        if len(self.move_history) >= self.total_actions:
            return 0, black_count, white_count

        return None, black_count, white_count

    def count_pieces(self) -> Tuple[int, int]:
        """Return (black_count, white_count) regardless of game state."""
        black_count = np.count_nonzero(self.board == BLACK)
        white_count = np.count_nonzero(self.board == WHITE)
        return black_count, white_count

    def get_board_state(self) -> np.ndarray:
        """Return a copy of the current board state."""
        return self.board.copy()

    def get_state_planes(self, player: Optional[int] = None) -> np.ndarray:
        """
        Return neural network input planes:
        Plane 0: current player's stones
        Plane 1: opponent's stones
        Plane 2: color to move (all 1s if black, all 0s if white)
        Shape: (3, board_size, board_size)
        """
        if player is None:
            player = self.current_player
        opponent = self.get_opponent(player)

        planes = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        planes[0] = (self.board == player).astype(np.float32)
        planes[1] = (self.board == opponent).astype(np.float32)
        planes[2] = 1.0 if player == BLACK else 0.0

        return planes

    def action_to_coords(self, action: int) -> Tuple[int, int]:
        """Convert action index to (row, col)."""
        return action // self.board_size, action % self.board_size

    def coords_to_action(self, row: int, col: int) -> int:
        """Convert (row, col) to action index."""
        return row * self.board_size + col

    def __str__(self) -> str:
        """ASCII board representation."""
        symbols = {EMPTY: '.', BLACK: 'X', WHITE: 'O'}
        lines = []
        lines.append("  " + " ".join(str(i % 10) for i in range(self.board_size)))
        for i in range(self.board_size):
            row_str = f"{i % 10:1d} " + " ".join(symbols[v] for v in self.board[i])
            lines.append(row_str)
        lines.append(f"Current: {'Black' if self.current_player == BLACK else 'White'}")
        return "\n".join(lines)


def play_game(black_policy, white_policy, verbose=False):
    """
    Play a full game using two policy functions.
    Each policy receives (game) and returns an action.
    Returns (winner, black_count, white_count, move_history).
    """
    game = GomokuGame()
    policies = {BLACK: black_policy, WHITE: white_policy}

    while not game.is_game_over():
        player = game.current_player
        policy = policies[player]
        action = policy(game)

        success, msg = game.make_move(action, player)
        if not success:
            raise ValueError(f"Policy returned invalid move: {action} for player {player}")

        if verbose:
            print(f"Player {'Black' if player == BLACK else 'White'}: {msg}")
            print(game)
            print()

    winner, black_count, white_count = game.get_winner()
    return winner, black_count, white_count, game.move_history
