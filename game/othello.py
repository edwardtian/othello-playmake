# Othello Game Engine
# Implements standard 8x8 Othello rules with strict validation.

import numpy as np
from typing import List, Tuple, Optional

# Board values
EMPTY = 0
BLACK = 1
WHITE = 2

# Action space: 0-63 are board squares, 64 is pass
PASS_ACTION = 64
BOARD_SIZE = 8
TOTAL_ACTIONS = 65

# Direction vectors for checking lines
directions = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),          (0, 1),
    (1, -1),  (1, 0),  (1, 1)
]


class OthelloGame:
    """
    Othello game state manager.
    The game engine only reports:
    - Whether a move is legal or not
    - When the game is over and which side wins
    No heuristic hints are provided to the AI.
    """

    def __init__(self):
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
        self.board[3, 3] = WHITE
        self.board[3, 4] = BLACK
        self.board[4, 3] = BLACK
        self.board[4, 4] = WHITE
        self.current_player = BLACK
        self.move_history = []
        self.pass_count = 0

    def reset(self):
        """Reset to initial state."""
        self.board.fill(EMPTY)
        self.board[3, 3] = WHITE
        self.board[3, 4] = BLACK
        self.board[4, 3] = BLACK
        self.board[4, 4] = WHITE
        self.current_player = BLACK
        self.move_history.clear()
        self.pass_count = 0
        return self

    def copy(self) -> 'OthelloGame':
        """Create a deep copy of the game state."""
        new_game = OthelloGame.__new__(OthelloGame)
        new_game.board = self.board.copy()
        new_game.current_player = self.current_player
        new_game.move_history = self.move_history.copy()
        new_game.pass_count = self.pass_count
        return new_game

    def get_opponent(self, player: int) -> int:
        """Return the opponent of the given player."""
        return WHITE if player == BLACK else BLACK

    def _in_bounds(self, row: int, col: int) -> bool:
        """Check if coordinates are within the board."""
        return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE

    def _get_flips(self, row: int, col: int, player: int) -> List[Tuple[int, int]]:
        """
        Get list of opponent stones that would be flipped by placing a stone at (row, col).
        Returns empty list if the move is invalid.
        """
        if not self._in_bounds(row, col) or self.board[row, col] != EMPTY:
            return []

        opponent = self.get_opponent(player)
        all_flips = []

        for dr, dc in directions:
            r, c = row + dr, col + dc
            line = []
            while self._in_bounds(r, c) and self.board[r, c] == opponent:
                line.append((r, c))
                r += dr
                c += dc
            if line and self._in_bounds(r, c) and self.board[r, c] == player:
                all_flips.extend(line)

        return all_flips

    def is_valid_move(self, action: int, player: Optional[int] = None) -> bool:
        """
        Check if an action is valid for the given player.
        action: 0-63 for board positions, 64 for pass.
        """
        if player is None:
            player = self.current_player

        if action == PASS_ACTION:
            # Pass is only valid if there are no legal moves
            return PASS_ACTION in self.get_legal_moves(player)

        if action < 0 or action >= PASS_ACTION:
            return False

        row, col = action // BOARD_SIZE, action % BOARD_SIZE
        return len(self._get_flips(row, col, player)) > 0

    def get_legal_moves(self, player: Optional[int] = None) -> List[int]:
        """Return a list of all legal action indices for the given player."""
        if player is None:
            player = self.current_player

        moves = []
        for action in range(PASS_ACTION):
            row, col = action // BOARD_SIZE, action % BOARD_SIZE
            if self._get_flips(row, col, player):
                moves.append(action)

        # If no legal moves, must pass
        if not moves:
            moves.append(PASS_ACTION)

        return moves

    def make_move(self, action: int, player: Optional[int] = None) -> Tuple[bool, str]:
        """
        Execute a move. Returns (success, message).
        If success is False, the board state is NOT modified.
        """
        if player is None:
            player = self.current_player

        if not self.is_valid_move(action, player):
            return False, f"Invalid move: action={action} for player={player}"

        if action == PASS_ACTION:
            self.move_history.append((action, player, []))
            self.current_player = self.get_opponent(player)
            self.pass_count += 1
            return True, "Pass"

        row, col = action // BOARD_SIZE, action % BOARD_SIZE
        flips = self._get_flips(row, col, player)

        # Place stone and flip opponent stones
        self.board[row, col] = player
        for r, c in flips:
            self.board[r, c] = player

        self.move_history.append((action, player, flips))
        self.current_player = self.get_opponent(player)
        self.pass_count = 0  # Reset pass count on a normal move

        return True, f"Move ({row},{col}) by player {player}, flipped {len(flips)} stones"

    def is_game_over(self) -> bool:
        """
        Game is over if:
        - Both players pass consecutively, OR
        - Board is full
        """
        if self.pass_count >= 2:
            return True
        if np.count_nonzero(self.board == EMPTY) == 0:
            return True
        return False

    def get_winner(self) -> Tuple[Optional[int], int, int]:
        """
        Returns (winner, black_count, white_count).
        winner: 1=BLACK wins, 2=WHITE wins, 0=DRAW, None=game not over
        """
        if not self.is_game_over():
            return None, 0, 0

        black_count = np.count_nonzero(self.board == BLACK)
        white_count = np.count_nonzero(self.board == WHITE)

        if black_count > white_count:
            return BLACK, black_count, white_count
        elif white_count > black_count:
            return WHITE, black_count, white_count
        else:
            return 0, black_count, white_count

    def get_board_state(self) -> np.ndarray:
        """Return a copy of the current board state."""
        return self.board.copy()

    def get_state_planes(self, player: Optional[int] = None) -> np.ndarray:
        """
        Return neural network input planes:
        Plane 0: current player's stones
        Plane 1: opponent's stones
        Plane 2: color to move (all 1s if black, all 0s if white)
        Shape: (3, 8, 8)
        """
        if player is None:
            player = self.current_player
        opponent = self.get_opponent(player)

        planes = np.zeros((3, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        planes[0] = (self.board == player).astype(np.float32)
        planes[1] = (self.board == opponent).astype(np.float32)
        planes[2] = 1.0 if player == BLACK else 0.0

        return planes

    def action_to_coords(self, action: int) -> Tuple[int, int]:
        """Convert action index to (row, col)."""
        if action == PASS_ACTION:
            return -1, -1
        return action // BOARD_SIZE, action % BOARD_SIZE

    def coords_to_action(self, row: int, col: int) -> int:
        """Convert (row, col) to action index."""
        return row * BOARD_SIZE + col

    def __str__(self) -> str:
        """ASCII board representation."""
        symbols = {EMPTY: '.', BLACK: 'X', WHITE: 'O'}
        lines = []
        lines.append("  0 1 2 3 4 5 6 7")
        for i in range(BOARD_SIZE):
            row_str = f"{i} " + " ".join(symbols[v] for v in self.board[i])
            lines.append(row_str)
        lines.append(f"Current: {'Black' if self.current_player == BLACK else 'White'}")
        return "\n".join(lines)


def play_game(black_policy, white_policy, verbose=False):
    """
    Play a full game using two policy functions.
    Each policy receives (game) and returns an action.
    Returns (winner, black_count, white_count, move_history).
    """
    game = OthelloGame()
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
