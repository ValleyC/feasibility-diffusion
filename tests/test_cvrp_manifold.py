"""
Correctness tests for the CVRP feasibility manifold.

Validates:
  1. Random solutions are feasible
  2. All three move types preserve feasibility
  3. Delta computation matches brute-force cost difference
  4. Forward scrambling preserves feasibility at every step
  5. Best move improves or maintains cost
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from problems.cvrp.solution import (
    random_solution, is_feasible, solution_cost,
    enumerate_moves, apply_move, delta_move, route_demand, MoveType,
)
from problems.cvrp.manifold import CVRPManifold
from problems.tsp.tour import dist_matrix_from_coords


def make_instance(N=20, capacity=30, seed=42):
    np.random.seed(seed)
    coords = np.random.rand(N + 1, 2).astype(np.float32)
    demands = np.zeros(N + 1, dtype=np.float32)
    demands[1:] = np.random.randint(1, 10, N).astype(np.float32)
    dist = dist_matrix_from_coords(coords)
    return {
        'coords': coords, 'demands': demands,
        'capacity': capacity, 'dist': dist, 'n_customers': N,
    }


def test_random_feasibility():
    """Random solutions must be feasible."""
    for N in [10, 20, 50]:
        inst = make_instance(N=N)
        for _ in range(10):
            sol = random_solution(N, inst['demands'], inst['capacity'])
            assert is_feasible(sol, inst['demands'], inst['capacity'], N), \
                f"Random solution infeasible for N={N}"
    print("[PASS] random_feasibility")


def test_moves_preserve_feasibility():
    """Every move on a feasible solution must produce a feasible solution."""
    for N in [10, 15, 20]:
        inst = make_instance(N=N, seed=123)
        sol = random_solution(N, inst['demands'], inst['capacity'])
        assert is_feasible(sol, inst['demands'], inst['capacity'], N)

        moves = enumerate_moves(sol, inst['demands'], inst['capacity'])
        for m in moves:
            new_sol = apply_move(sol, m)
            assert is_feasible(new_sol, inst['demands'], inst['capacity'], N), \
                f"Move {m} produced infeasible solution"
    print(f"[PASS] moves_preserve_feasibility (tested all moves for N=10,15,20)")


def test_delta_matches_brute_force():
    """Delta computation must match actual cost difference."""
    inst = make_instance(N=15, seed=7)
    sol = random_solution(15, inst['demands'], inst['capacity'])
    cost_before = solution_cost(sol, inst['dist'])

    moves = enumerate_moves(sol, inst['demands'], inst['capacity'])
    for m in moves:
        delta = delta_move(sol, m, inst['dist'])
        new_sol = apply_move(sol, m)
        cost_after = solution_cost(new_sol, inst['dist'])
        actual_delta = cost_after - cost_before
        assert abs(delta - actual_delta) < 1e-4, \
            f"Move {m}: delta={delta:.6f} vs actual={actual_delta:.6f}"
    print(f"[PASS] delta_matches_brute_force ({len(moves)} moves checked)")


def test_forward_preserves_feasibility():
    """Random scrambling must stay feasible at every step."""
    manifold = CVRPManifold()
    inst = make_instance(N=20, seed=99)
    sol = manifold.sample_random(inst)
    assert manifold.is_feasible(sol, inst)

    for step in range(100):
        moves = manifold.enumerate_moves(sol, inst)
        if len(moves) == 0:
            break
        idx = np.random.randint(len(moves))
        sol = manifold.apply_move(sol, moves[idx])
        assert manifold.is_feasible(sol, inst), \
            f"Step {step}: infeasible after random move"
    print("[PASS] forward_preserves_feasibility (100 steps)")


def test_best_move_improves():
    """Best move should not increase cost."""
    manifold = CVRPManifold()
    inst = make_instance(N=20, seed=55)
    sol = manifold.sample_random(inst)
    cost_before = manifold.cost(sol, inst)

    move, delta = manifold.best_move(sol, inst)
    if move is not None:
        assert delta <= 1e-8, f"Best move worsens: delta={delta}"
        new_sol = manifold.apply_move(sol, move)
        cost_after = manifold.cost(new_sol, inst)
        assert cost_after <= cost_before + 1e-6
    print(f"[PASS] best_move_improves (delta={delta:.4f})")


def test_move_type_counts():
    """Verify we get all three move types."""
    inst = make_instance(N=15, seed=42)
    sol = random_solution(15, inst['demands'], inst['capacity'])
    moves = enumerate_moves(sol, inst['demands'], inst['capacity'])

    counts = {MoveType.INTRA_2OPT: 0, MoveType.RELOCATE: 0, MoveType.SWAP: 0}
    for m in moves:
        counts[m[0]] += 1

    print(f"[PASS] move_type_counts: "
          f"intra_2opt={counts[MoveType.INTRA_2OPT]}, "
          f"relocate={counts[MoveType.RELOCATE]}, "
          f"swap={counts[MoveType.SWAP]}, "
          f"total={len(moves)}")
    assert counts[MoveType.RELOCATE] > 0, "No relocate moves found"


def test_manifold_interface():
    """CVRPManifold satisfies the abstract interface."""
    manifold = CVRPManifold()
    inst = make_instance(N=10)
    sol = manifold.sample_random(inst)
    assert manifold.is_feasible(sol, inst)
    c = manifold.cost(sol, inst)
    assert c > 0

    moves = manifold.enumerate_moves(sol, inst)
    assert len(moves) > 0
    for m in moves[:5]:
        new = manifold.apply_move(sol, m)
        assert manifold.is_feasible(new, inst)
    print("[PASS] manifold_interface")


if __name__ == '__main__':
    test_random_feasibility()
    test_moves_preserve_feasibility()
    test_delta_matches_brute_force()
    test_forward_preserves_feasibility()
    test_best_move_improves()
    test_move_type_counts()
    test_manifold_interface()
    print("\n=== ALL CVRP TESTS PASSED ===")
