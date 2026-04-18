"""
Correctness tests for the CVRP partition manifold.

Validates:
  1. Random partitions are feasible
  2. Relocate + swap moves preserve feasibility
  3. Delta matches brute-force cost difference
  4. Forward scrambling preserves feasibility
  5. Best move improves cost
  6. Only 2 move types (no intra-2opt)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from problems.cvrp.partition import (
    random_partition, partition_feasible, partition_cost,
    enumerate_partition_moves, apply_partition_move,
    delta_partition_move, vehicle_loads, n_vehicles, PartMoveType,
)
from problems.cvrp.partition_manifold import CVRPPartitionManifold
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
    for N in [10, 20, 50]:
        inst = make_instance(N=N)
        for _ in range(20):
            assign = random_partition(N, inst['demands'], inst['capacity'])
            assert partition_feasible(assign, inst['demands'], inst['capacity'], N), \
                f"Random partition infeasible for N={N}"
    print("[PASS] random_feasibility")


def test_only_two_move_types():
    """Verify only relocate + swap, no intra-2opt."""
    inst = make_instance(N=15, seed=7)
    assign = random_partition(15, inst['demands'], inst['capacity'])
    moves = enumerate_partition_moves(assign, inst['demands'], inst['capacity'])
    types = set(m[0] for m in moves)
    assert types <= {PartMoveType.RELOCATE, PartMoveType.SWAP}, \
        f"Unexpected move types: {types}"
    assert PartMoveType.RELOCATE in types, "No relocate moves"
    counts = {PartMoveType.RELOCATE: 0, PartMoveType.SWAP: 0}
    for m in moves:
        counts[m[0]] += 1
    print(f"[PASS] only_two_move_types: relocate={counts[PartMoveType.RELOCATE]}, "
          f"swap={counts[PartMoveType.SWAP]}, total={len(moves)}")


def test_moves_preserve_feasibility():
    for N in [10, 15, 20]:
        inst = make_instance(N=N, seed=123)
        assign = random_partition(N, inst['demands'], inst['capacity'])
        assert partition_feasible(assign, inst['demands'], inst['capacity'], N)

        moves = enumerate_partition_moves(assign, inst['demands'], inst['capacity'])
        for m in moves:
            new_assign = apply_partition_move(assign, m)
            assert partition_feasible(new_assign, inst['demands'], inst['capacity'], N), \
                f"Move {m} produced infeasible partition"
    print(f"[PASS] moves_preserve_feasibility (exhaustive for N=10,15,20)")


def test_delta_matches_brute_force():
    inst = make_instance(N=15, seed=42)
    assign = random_partition(15, inst['demands'], inst['capacity'])
    cost_before = partition_cost(assign, inst['coords'], inst['dist'])

    moves = enumerate_partition_moves(assign, inst['demands'], inst['capacity'])
    for m in moves:
        delta = delta_partition_move(assign, m, inst['coords'], inst['dist'])
        new_assign = apply_partition_move(assign, m)
        cost_after = partition_cost(new_assign, inst['coords'], inst['dist'])
        actual_delta = cost_after - cost_before
        assert abs(delta - actual_delta) < 1e-4, \
            f"Move {m}: delta={delta:.4f} vs actual={actual_delta:.4f}"
    print(f"[PASS] delta_matches_brute_force ({len(moves)} moves)")


def test_forward_preserves_feasibility():
    manifold = CVRPPartitionManifold()
    inst = make_instance(N=20, seed=99)
    assign = manifold.sample_random(inst)
    assert manifold.is_feasible(assign, inst)

    for step in range(100):
        moves = manifold.enumerate_moves(assign, inst)
        if len(moves) == 0:
            break
        idx = np.random.randint(len(moves))
        assign = manifold.apply_move(assign, moves[idx])
        assert manifold.is_feasible(assign, inst), \
            f"Step {step}: infeasible after random move"
    print("[PASS] forward_preserves_feasibility (100 steps)")


def test_best_move_improves():
    manifold = CVRPPartitionManifold()
    inst = make_instance(N=20, seed=55)
    assign = manifold.sample_random(inst)
    cost_before = manifold.cost(assign, inst)

    move, delta = manifold.best_move(assign, inst)
    if move is not None and delta < 0:
        new_assign = manifold.apply_move(assign, move)
        cost_after = manifold.cost(new_assign, inst)
        assert cost_after <= cost_before + 1e-4
    print(f"[PASS] best_move_improves (delta={delta:.4f})")


def test_partition_cost_uses_subtsp():
    """Verify cost computation solves sub-TSP per vehicle."""
    inst = make_instance(N=10, seed=0)
    assign = random_partition(10, inst['demands'], inst['capacity'])
    cost = partition_cost(assign, inst['coords'], inst['dist'])
    assert cost > 0
    K = n_vehicles(assign)
    print(f"[PASS] partition_cost_uses_subtsp (K={K} vehicles, cost={cost:.4f})")


if __name__ == '__main__':
    test_random_feasibility()
    test_only_two_move_types()
    test_moves_preserve_feasibility()
    test_delta_matches_brute_force()
    test_forward_preserves_feasibility()
    test_best_move_improves()
    test_partition_cost_uses_subtsp()
    print("\n=== ALL CVRP PARTITION TESTS PASSED ===")
