"""
Correctness tests for the TSP feasibility manifold.

Validates:
  1. Tour validity (permutation invariant)
  2. 2-opt move count: exactly N(N-3)/2
  3. 2-opt preserves feasibility (every move produces a valid tour)
  4. Delta computation matches brute-force cost difference
  5. Forward scrambling preserves feasibility at every step
  6. Scrambling increases cost on average (entropy increases)
  7. Best-move greedy improves or maintains cost
  8. 2-opt solver produces valid, reasonable tours
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from problems.tsp.tour import (
    random_tour, is_valid_tour, tour_cost, enumerate_2opt,
    apply_2opt, delta_2opt, dist_matrix_from_coords, tour_cost_coords,
)
from problems.tsp.manifold import TSPManifold
from problems.tsp.data import generate_instance, solve_2opt
from core.forward import scramble, scramble_trajectory, compute_move_labels


def test_tour_validity():
    """Random tours must be valid permutations."""
    for N in [5, 10, 20, 50]:
        for _ in range(10):
            tour = random_tour(N)
            assert is_valid_tour(tour), f"Invalid tour for N={N}: {tour}"
    print("[PASS] tour_validity")


def test_2opt_move_count():
    """Number of valid 2-opt moves must be N(N-3)/2."""
    for N in [5, 10, 20, 50, 100]:
        moves = enumerate_2opt(N)
        expected = N * (N - 3) // 2
        assert len(moves) == expected, \
            f"N={N}: expected {expected} moves, got {len(moves)}"
    print("[PASS] 2opt_move_count")


def test_2opt_preserves_feasibility():
    """Every 2-opt move on a valid tour must produce a valid tour."""
    for N in [5, 10, 20]:
        tour = random_tour(N)
        assert is_valid_tour(tour)
        moves = enumerate_2opt(N)
        for i, j in moves:
            new_tour = apply_2opt(tour, i, j)
            assert is_valid_tour(new_tour), \
                f"2-opt({i},{j}) on {tour} produced invalid: {new_tour}"
    print("[PASS] 2opt_preserves_feasibility")


def test_delta_matches_brute_force():
    """Delta computation must match actual cost difference."""
    np.random.seed(42)
    for N in [5, 10, 20]:
        coords = np.random.rand(N, 2)
        dist = dist_matrix_from_coords(coords)
        tour = random_tour(N)
        cost_before = tour_cost(tour, dist)
        moves = enumerate_2opt(N)
        for i, j in moves:
            delta = delta_2opt(tour, i, j, dist)
            new_tour = apply_2opt(tour, i, j)
            cost_after = tour_cost(new_tour, dist)
            actual_delta = cost_after - cost_before
            assert abs(delta - actual_delta) < 1e-8, \
                f"N={N}, move=({i},{j}): delta={delta:.8f} vs actual={actual_delta:.8f}"
    print("[PASS] delta_matches_brute_force")


def test_tour_cost_consistency():
    """tour_cost with dist_matrix must match tour_cost_coords."""
    np.random.seed(0)
    for N in [5, 10, 20]:
        coords = np.random.rand(N, 2).astype(np.float32)
        dist = dist_matrix_from_coords(coords)
        tour = random_tour(N)
        c1 = tour_cost(tour, dist)
        c2 = tour_cost_coords(tour, coords)
        assert abs(c1 - c2) < 1e-4, f"Cost mismatch: {c1} vs {c2}"
    print("[PASS] tour_cost_consistency")


def test_forward_preserves_feasibility():
    """Every step of the forward scrambling must be feasible."""
    np.random.seed(123)
    manifold = TSPManifold()
    for N in [10, 20, 50]:
        coords = np.random.rand(N, 2)
        dist = dist_matrix_from_coords(coords)
        tour = random_tour(N)
        trajectory = scramble_trajectory(manifold, tour, dist, n_steps=100)
        for t, s in enumerate(trajectory):
            assert manifold.is_feasible(s, dist), \
                f"N={N}, step {t}: infeasible state {s}"
    print("[PASS] forward_preserves_feasibility")


def test_scrambling_increases_cost():
    """On average, scrambling a good tour should increase cost."""
    np.random.seed(99)
    manifold = TSPManifold()
    N = 20
    coords = np.random.rand(N, 2)
    dist = dist_matrix_from_coords(coords)
    opt_tour, opt_cost = solve_2opt(coords, max_restarts=N)

    scrambled_costs = []
    for _ in range(50):
        s = scramble(manifold, opt_tour, dist, n_steps=50)
        scrambled_costs.append(manifold.cost(s, dist))

    avg_scrambled = np.mean(scrambled_costs)
    assert avg_scrambled > opt_cost, \
        f"Scrambled cost ({avg_scrambled:.4f}) should be > optimal ({opt_cost:.4f})"
    print(f"[PASS] scrambling_increases_cost "
          f"(opt={opt_cost:.4f}, avg_scrambled={avg_scrambled:.4f}, "
          f"ratio={avg_scrambled/opt_cost:.2f}x)")


def test_best_move_improves():
    """Greedy best-move should not increase cost (on a non-optimal tour)."""
    np.random.seed(7)
    manifold = TSPManifold()
    N = 20
    coords = np.random.rand(N, 2)
    dist = dist_matrix_from_coords(coords)
    tour = random_tour(N)  # start from random (likely improvable)
    cost_before = manifold.cost(tour, dist)

    move, delta = manifold.best_move(tour, dist)
    assert delta <= 1e-10, \
        f"Best move should not worsen or be neutral: delta={delta}"

    new_tour = manifold.apply_move(tour, move)
    cost_after = manifold.cost(new_tour, dist)
    assert cost_after <= cost_before + 1e-8, \
        f"Cost increased after best move: {cost_before:.6f} -> {cost_after:.6f}"
    print(f"[PASS] best_move_improves "
          f"(before={cost_before:.4f}, after={cost_after:.4f})")


def test_move_labels():
    """compute_move_labels must return correct deltas and valid best move."""
    np.random.seed(42)
    manifold = TSPManifold()
    N = 15
    coords = np.random.rand(N, 2)
    dist = dist_matrix_from_coords(coords)
    tour = random_tour(N)

    moves, deltas = compute_move_labels(manifold, tour, dist)
    assert len(moves) == manifold.num_moves(N)
    assert len(deltas) == len(moves)

    # Best move from labels should match manifold.best_move
    best_idx = np.argmin(deltas)
    best_move_label = moves[best_idx]
    best_move_manifold, best_delta_manifold = manifold.best_move(tour, dist)
    assert abs(deltas[best_idx] - best_delta_manifold) < 1e-8, \
        f"Label best delta {deltas[best_idx]} != manifold best delta {best_delta_manifold}"
    print("[PASS] move_labels")


def test_solver_quality():
    """2-opt solver should produce tours better than random."""
    np.random.seed(0)
    N = 20
    coords = np.random.rand(N, 2)
    dist = dist_matrix_from_coords(coords)

    # Random baseline
    random_costs = [tour_cost(random_tour(N), dist) for _ in range(100)]
    avg_random = np.mean(random_costs)

    # 2-opt solver
    opt_tour, opt_cost = solve_2opt(coords, max_restarts=N)
    assert is_valid_tour(opt_tour)
    assert opt_cost < avg_random, \
        f"Solver ({opt_cost:.4f}) should beat random avg ({avg_random:.4f})"
    print(f"[PASS] solver_quality "
          f"(solver={opt_cost:.4f}, random_avg={avg_random:.4f}, "
          f"ratio={avg_random/opt_cost:.2f}x)")


def test_manifold_interface():
    """TSPManifold must satisfy all abstract interface contracts."""
    np.random.seed(0)
    manifold = TSPManifold()
    N = 10
    coords = np.random.rand(N, 2)
    dist = dist_matrix_from_coords(coords)

    s = manifold.sample_random(dist)
    assert manifold.is_feasible(s, dist)
    c = manifold.cost(s, dist)
    assert isinstance(c, float) and c > 0

    moves = manifold.enumerate_moves(s, dist)
    assert len(moves) == manifold.num_moves(N)

    for m in moves[:5]:
        s2 = manifold.apply_move(s, m)
        assert manifold.is_feasible(s2, dist)
        d = manifold.move_delta(s, m, dist)
        assert isinstance(d, float)

    print("[PASS] manifold_interface")


if __name__ == '__main__':
    test_tour_validity()
    test_2opt_move_count()
    test_2opt_preserves_feasibility()
    test_delta_matches_brute_force()
    test_tour_cost_consistency()
    test_forward_preserves_feasibility()
    test_scrambling_increases_cost()
    test_best_move_improves()
    test_move_labels()
    test_solver_quality()
    test_manifold_interface()
    print("\n=== ALL TESTS PASSED ===")
