"""
Forward Process: CTMC-based Scrambling on the Feasibility Manifold.

The forward process takes a "clean" solution s* and progressively corrupts it
by applying random valid moves (e.g., random 2-opt swaps for TSP).

Formally, this is a continuous-time Markov chain (CTMC) on the graph
G = (S, E) induced by the neighborhood structure of the manifold:

    Rate matrix Q_t(s, s') = β(t)  if (s, s') ∈ E
                            = 0     otherwise
    Q_t(s, s) = -β(t) · |N(s)|

where β(t) is the noise schedule and |N(s)| is the number of valid moves
from state s.

Key properties:
  1. FEASIBILITY PRESERVED: every transition goes to a valid neighbor,
     so s_t is feasible for all t.
  2. ERGODICITY: if G is connected (guaranteed for TSP 2-opt), the chain
     mixes to the uniform distribution over S as t → ∞.
  3. REVERSIBILITY: the time-reversed CTMC exists and can be parameterized
     by a neural network (this is the reverse/denoising process).

In practice, we discretize: at each step, apply one random valid move.
The number of steps T controls the noise level. For training, we sample
T ~ Uniform(1, T_max) and produce (s_T, T, s*) triples.

Noise schedule β(t):
  - In discrete simulation, β is implicit — one move per step.
  - T_max should be ≥ mixing time of the chain.
  - For TSP 2-opt: mixing time is O(N^4 log N) (Dyer, Frieze & Kannan, 1994),
    but empirically O(N^2) random swaps suffice for good scrambling.
"""

import numpy as np
from typing import Tuple, List, Optional

from core.manifold import FeasibilityManifold


def scramble(manifold: FeasibilityManifold,
             solution: np.ndarray,
             instance: np.ndarray,
             n_steps: int) -> np.ndarray:
    """Apply n_steps random valid moves to scramble a solution.

    This is the discrete-time forward process: s_0 = solution, and
    s_{t+1} = apply_random_move(s_t) for t = 0, ..., n_steps-1.

    INVARIANT: output is feasible if input is feasible (by induction,
    since each move preserves feasibility).

    Args:
        manifold: the feasibility manifold
        solution: initial feasible solution (e.g., an optimal tour)
        instance: problem instance (e.g., distance matrix)
        n_steps: number of random moves to apply

    Returns:
        Scrambled solution after n_steps random valid moves.
    """
    s = solution.copy()
    for _ in range(n_steps):
        s = manifold.apply_random_move(s, instance)
    return s


def scramble_trajectory(manifold: FeasibilityManifold,
                        solution: np.ndarray,
                        instance: np.ndarray,
                        n_steps: int) -> List[np.ndarray]:
    """Apply n_steps random moves, returning the full trajectory.

    Returns:
        List of n_steps+1 solutions: [s_0, s_1, ..., s_{n_steps}]
        where s_0 = solution (the clean input).
    """
    trajectory = [solution.copy()]
    s = solution.copy()
    for _ in range(n_steps):
        s = manifold.apply_random_move(s, instance)
        trajectory.append(s.copy())
    return trajectory


def sample_training_pair(manifold: FeasibilityManifold,
                         clean_solution: np.ndarray,
                         instance: np.ndarray,
                         t_max: int,
                         t: Optional[int] = None,
                         ) -> Tuple[np.ndarray, int, np.ndarray]:
    """Sample a (noisy_solution, noise_level, clean_solution) training triple.

    Used for denoising score matching: given s_t (noisy at level t),
    the model must predict which move reduces noise (= moves toward s*).

    Args:
        manifold: the feasibility manifold
        clean_solution: the "target" (e.g., optimal or near-optimal tour)
        instance: problem instance
        t_max: maximum noise level (number of scrambling steps)
        t: if None, sample t ~ Uniform(1, t_max). If given, use that level.

    Returns:
        (noisy_solution, t, clean_solution)
    """
    if t is None:
        t = np.random.randint(1, t_max + 1)
    noisy = scramble(manifold, clean_solution, instance, t)
    return noisy, t, clean_solution


def sample_training_batch(manifold: FeasibilityManifold,
                          clean_solutions: List[np.ndarray],
                          instances: List[np.ndarray],
                          t_max: int,
                          batch_size: int,
                          ) -> List[Tuple[np.ndarray, int, np.ndarray, np.ndarray]]:
    """Sample a batch of training triples.

    Returns:
        List of (noisy_solution, t, clean_solution, instance) tuples.
    """
    batch = []
    for _ in range(batch_size):
        idx = np.random.randint(len(clean_solutions))
        noisy, t, clean = sample_training_pair(
            manifold, clean_solutions[idx], instances[idx], t_max
        )
        batch.append((noisy, t, clean, instances[idx]))
    return batch


def compute_move_labels(manifold: FeasibilityManifold,
                        noisy_solution: np.ndarray,
                        instance: np.ndarray,
                        ) -> Tuple[List, np.ndarray]:
    """For each valid move, compute the cost change (delta).

    The "label" for supervised training is the move with the most
    negative delta (= biggest improvement). This is the greedy-optimal
    single-step denoising action.

    Returns:
        moves: list of move descriptors
        deltas: (M,) array of cost changes, one per move
    """
    moves = manifold.enumerate_moves(noisy_solution, instance)
    deltas = np.array([
        manifold.move_delta(noisy_solution, m, instance) for m in moves
    ])
    return moves, deltas
