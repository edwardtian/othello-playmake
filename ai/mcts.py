"""
Monte Carlo Tree Search (MCTS) for Othello.

Implements AlphaZero-style MCTS with:
  - UCT selection
  - Neural network guided expansion
  - Dirichlet noise at root for exploration
  - Temperature-based move selection
"""

import math
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from game.othello import PASS_ACTION


class MCTSNode:
    """A node in the MCTS tree."""

    def __init__(self, prior: float = 0.0, action: int = -1, parent: Optional['MCTSNode'] = None):
        self.prior = prior          # Prior probability from neural net
        self.action = action        # Action that led to this node
        self.parent = parent
        self.children: Dict[int, 'MCTSNode'] = {}
        self.visit_count = 0
        self.value_sum = 0.0        # Sum of values from all visits
        self.is_expanded = False

    @property
    def q_value(self) -> float:
        """Average value (Q) of this node."""
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def select_child(self, c_puct: float) -> Tuple[int, 'MCTSNode']:
        """
        Select child with highest UCT score.
        UCT = Q + c_puct * P * sqrt(N_parent) / (1 + N_child)
        """
        best_score = -float('inf')
        best_action = -1
        best_child = None

        sqrt_parent_visits = math.sqrt(self.visit_count)

        for action, child in self.children.items():
            # UCT formula
            if child.visit_count > 0:
                q = -child.q_value  # Negate because value is from parent's perspective
            else:
                q = 0.0

            u = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
            score = q + u

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, action_priors: np.ndarray, valid_actions: List[int]):
        """Expand leaf node with prior probabilities for valid actions."""
        for action in valid_actions:
            if action not in self.children:
                self.children[action] = MCTSNode(
                    prior=action_priors[action],
                    action=action,
                    parent=self
                )
        self.is_expanded = True

    def backup(self, value: float):
        """Backup value up the tree."""
        self.visit_count += 1
        self.value_sum += value
        if self.parent is not None:
            # Value is from perspective of player who just moved, so negate for parent
            self.parent.backup(-value)


class MCTS:
    """
    Monte Carlo Tree Search using a policy-value network.
    """

    def __init__(
        self,
        model,
        num_simulations: int = 800,
        c_puct: float = 1.5,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        action_size: int = None,
    ):
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.action_size = action_size or getattr(model, 'action_size', 65)

    @torch.no_grad()
    def search(self, game: object, temperature: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run MCTS from the given game state.
        
        Args:
            game: Current OthelloGame state
            temperature: Temperature for move selection (1.0 = uniform, 0.0 = argmax)
        
        Returns:
            action_probs: (action_size,) visit count distribution
            root_value: scalar value estimate from root
        """
        # Get initial state and valid moves
        state_planes = game.get_state_planes()
        valid_actions = game.get_legal_moves()

        # Create root node
        root = MCTSNode()

        # Evaluate root with neural network
        state_tensor = torch.from_numpy(state_planes).float().unsqueeze(0).to(next(self.model.parameters()).device)
        policy_logits, value = self.model.forward(state_tensor)
        policy = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()

        # Mask invalid actions and renormalize
        masked_policy = np.zeros_like(policy)
        masked_policy[valid_actions] = policy[valid_actions]
        policy_sum = masked_policy.sum()
        if policy_sum > 0:
            masked_policy /= policy_sum
        else:
            # Fallback to uniform over valid actions
            masked_policy[valid_actions] = 1.0 / len(valid_actions)

        # Add Dirichlet noise at root for exploration
        if self.dirichlet_epsilon > 0 and len(valid_actions) > 1:
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(valid_actions))
            for i, action in enumerate(valid_actions):
                masked_policy[action] = (
                    (1 - self.dirichlet_epsilon) * masked_policy[action]
                    + self.dirichlet_epsilon * noise[i]
                )
            # Renormalize
            masked_policy /= masked_policy.sum()

        root.expand(masked_policy, valid_actions)
        root_value = value.item()

        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_game = game.copy()
            path = [node]

            # Selection: traverse tree until leaf
            while node.is_expanded and node.children:
                action, node = node.select_child(self.c_puct)
                search_game.make_move(action)
                path.append(node)

            # Evaluation
            if search_game.is_game_over():
                # Terminal state: actual outcome
                winner, _, _ = search_game.get_winner()
                if winner == 0:
                    value = 0.0
                else:
                    # Value from perspective of current player in the original game
                    # But search_game.current_player is the player to move in the terminal state
                    # The winner is from absolute perspective
                    # We want value from perspective of the player who just moved (parent)
                    # Actually, in backup we negate at each level, so we need value from
                    # perspective of current node (player to move at leaf)
                    value = 1.0 if winner == search_game.current_player else -1.0
            else:
                # Evaluate with neural network
                leaf_state = search_game.get_state_planes()
                leaf_tensor = torch.from_numpy(leaf_state).float().unsqueeze(0).to(next(self.model.parameters()).device)
                policy_logits, value_tensor = self.model.forward(leaf_tensor)
                leaf_policy = torch.softmax(policy_logits, dim=-1).squeeze(0).cpu().numpy()
                value = value_tensor.item()

                # Expand leaf
                leaf_valid_actions = search_game.get_legal_moves()
                masked_leaf_policy = np.zeros_like(leaf_policy)
                masked_leaf_policy[leaf_valid_actions] = leaf_policy[leaf_valid_actions]
                policy_sum = masked_leaf_policy.sum()
                if policy_sum > 0:
                    masked_leaf_policy /= policy_sum
                else:
                    masked_leaf_policy[leaf_valid_actions] = 1.0 / len(leaf_valid_actions)

                node.expand(masked_leaf_policy, leaf_valid_actions)

            # Backup
            # Value is from perspective of player to move at leaf
            # We negate at each parent level
            for i, node in enumerate(reversed(path)):
                if i == 0:
                    node.backup(value)
                # Note: backup already handles negation recursively

        # Extract visit counts for root children
        action_probs = np.zeros(self.action_size, dtype=np.float32)
        for action, child in root.children.items():
            action_probs[action] = child.visit_count

        # Apply temperature
        if temperature == 0.0:
            # Argmax
            best_action = np.argmax(action_probs)
            action_probs.fill(0)
            action_probs[best_action] = 1.0
        else:
            # Smooth with temperature
            action_probs = action_probs ** (1.0 / temperature)
            action_probs_sum = action_probs.sum()
            if action_probs_sum > 0:
                action_probs /= action_probs_sum
            else:
                # Fallback to uniform
                action_probs[valid_actions] = 1.0 / len(valid_actions)

        return action_probs, np.array([root_value], dtype=np.float32)

    def get_best_move(self, game: object, temperature: float = 0.0) -> int:
        """Run MCTS and return the best action."""
        action_probs, _ = self.search(game, temperature=temperature)
        return int(np.argmax(action_probs))
