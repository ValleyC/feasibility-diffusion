"""
Abstract Feasibility Manifold for Combinatorial Optimization.

A feasibility manifold M defines:
  - A discrete state space S of feasible solutions (e.g., all Hamiltonian cycles)
  - A neighborhood structure N(s) for each state s (e.g., all 2-opt neighbors)
  - A cost function c(s) (e.g., tour length)

The neighborhood structure induces a graph G = (S, E) where (s, s') in E iff
s' in N(s). A CTMC on this graph defines our diffusion process:
  - Forward: random walk on G scrambles good solutions toward uniformity
  - Reverse: learned denoising walks back toward optimal solutions

Key theoretical property: G must be CONNECTED (any state reachable from any
other via a sequence of moves). This ensures the CTMC is ergodic and mixes
to a unique stationary distribution.

For TSP with 2-opt: connectivity is guaranteed (Lin, 1965).
For CVRP with feasible swaps: connectivity holds when capacity slack exists.
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Any
import numpy as np


class FeasibilityManifold(ABC):
    """Abstract base for a feasibility manifold over CO solutions."""

    @abstractmethod
    def sample_random(self, instance: Any) -> np.ndarray:
        """Sample a uniformly random feasible solution for the given instance.

        Args:
            instance: problem instance (e.g., distance matrix for TSP)
        Returns:
            A feasible solution (representation depends on subclass)
        """

    @abstractmethod
    def cost(self, solution: np.ndarray, instance: Any) -> float:
        """Compute the objective value of a feasible solution.

        Args:
            solution: a feasible solution
            instance: problem instance
        Returns:
            Objective value (lower is better)
        """

    @abstractmethod
    def is_feasible(self, solution: np.ndarray, instance: Any) -> bool:
        """Check whether a solution is feasible.

        This is used as a correctness invariant — every state in our
        diffusion process must pass this check.
        """

    @abstractmethod
    def enumerate_moves(self, solution: np.ndarray, instance: Any) -> List[Any]:
        """Enumerate all valid moves from the current solution.

        Each move transforms the solution to a neighboring feasible solution.
        The set of moves defines the neighborhood N(s).

        Returns:
            List of move descriptors (interpretation depends on subclass)
        """

    @abstractmethod
    def apply_move(self, solution: np.ndarray, move: Any) -> np.ndarray:
        """Apply a move to produce a new feasible solution.

        INVARIANT: if is_feasible(solution), then is_feasible(apply_move(solution, move))
        for any move in enumerate_moves(solution).

        Args:
            solution: current feasible solution
            move: a move descriptor from enumerate_moves()
        Returns:
            New feasible solution (must not modify the input)
        """

    @abstractmethod
    def move_delta(self, solution: np.ndarray, move: Any, instance: Any) -> float:
        """Compute the change in cost from applying a move WITHOUT applying it.

        delta = cost(apply_move(s, m)) - cost(s)
        Negative delta = improvement.

        This must be efficient (O(1) for 2-opt) since we evaluate all moves
        during training to find the best one.
        """

    def apply_random_move(self, solution: np.ndarray, instance: Any) -> np.ndarray:
        """Apply a uniformly random valid move. Used in the forward process."""
        moves = self.enumerate_moves(solution, instance)
        idx = np.random.randint(len(moves))
        return self.apply_move(solution, moves[idx])

    def best_move(self, solution: np.ndarray, instance: Any) -> Tuple[Any, float]:
        """Find the single move giving the largest cost reduction.

        Returns:
            (best_move, best_delta) where best_delta < 0 means improvement.
            If no improving move exists, returns the least-worsening move.
        """
        moves = self.enumerate_moves(solution, instance)
        best_m, best_d = None, float('inf')
        for m in moves:
            d = self.move_delta(solution, m, instance)
            if d < best_d:
                best_d = d
                best_m = m
        return best_m, best_d

    def num_moves(self, N: int) -> int:
        """Number of valid moves for a problem of size N (for pre-allocation)."""
        raise NotImplementedError
