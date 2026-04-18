"""
Unified test: verify feasibility preservation across ALL problem manifolds.
Tests the core invariant: every move on a feasible state produces a feasible state.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from problems.tsp.tour import dist_matrix_from_coords


def test_manifold(name, manifold, instance, n_scramble=50):
    """Generic test for any FeasibilityManifold implementation."""
    # 1. Random solution is feasible
    sol = manifold.sample_random(instance)
    assert manifold.is_feasible(sol, instance), f"{name}: random solution infeasible"

    # 2. All moves preserve feasibility
    moves = manifold.enumerate_moves(sol, instance)
    assert len(moves) > 0, f"{name}: no moves available"
    for m in moves:
        new = manifold.apply_move(sol, m)
        assert manifold.is_feasible(new, instance), \
            f"{name}: move {m} produced infeasible state"

    # 3. Delta matches brute force (spot check first 20 moves)
    cost_before = manifold.cost(sol, instance)
    for m in moves[:20]:
        delta = manifold.move_delta(sol, m, instance)
        new = manifold.apply_move(sol, m)
        actual = manifold.cost(new, instance) - cost_before
        assert abs(delta - actual) < 1e-3, \
            f"{name}: delta {delta:.4f} != actual {actual:.4f} for move {m}"

    # 4. Forward scrambling stays feasible
    s = sol
    for step in range(n_scramble):
        mv = manifold.enumerate_moves(s, instance)
        if len(mv) == 0:
            break
        s = manifold.apply_move(s, mv[np.random.randint(len(mv))])
        assert manifold.is_feasible(s, instance), \
            f"{name}: step {step} infeasible"

    # 5. Best move is consistent (apply it and check delta matches)
    move, delta = manifold.best_move(sol, instance)
    if move is not None:
        new = manifold.apply_move(sol, move)
        actual_delta = manifold.cost(new, instance) - cost_before
        assert abs(delta - actual_delta) < 1e-3, \
            f"{name}: best move delta inconsistent: {delta:.4f} vs {actual_delta:.4f}"

    print(f"[PASS] {name}: {len(moves)} moves, {n_scramble} scramble steps OK")


def make_tsp():
    from problems.tsp.manifold import TSPManifold
    N = 15
    np.random.seed(0)
    coords = np.random.rand(N, 2).astype(np.float32)
    dist = dist_matrix_from_coords(coords)
    return "TSP", TSPManifold(), dist


def make_cvrp():
    from problems.cvrp.partition_manifold import CVRPPartitionManifold
    N = 15
    np.random.seed(1)
    coords = np.random.rand(N + 1, 2).astype(np.float32)
    demands = np.zeros(N + 1, dtype=np.float32)
    demands[1:] = np.random.randint(1, 10, N).astype(np.float32)
    dist = dist_matrix_from_coords(coords)
    return "CVRP", CVRPPartitionManifold(), {
        'coords': coords, 'demands': demands, 'capacity': 30.0,
        'dist': dist, 'n_customers': N,
    }


def make_pctsp():
    from problems.pctsp.manifold import PCTSPManifold
    N = 15
    np.random.seed(2)
    coords = np.random.rand(N + 1, 2).astype(np.float32)
    prizes = np.zeros(N + 1, dtype=np.float32)
    prizes[1:] = np.random.uniform(0.1, 1.0, N).astype(np.float32)
    penalties = np.zeros(N + 1, dtype=np.float32)
    penalties[1:] = np.random.uniform(0.1, 0.5, N).astype(np.float32)
    dist = dist_matrix_from_coords(coords)
    return "PCTSP", PCTSPManifold(), {
        'coords': coords, 'prizes': prizes, 'penalties': penalties,
        'min_prize': float(prizes.sum() * 0.5), 'dist': dist, 'n_customers': N,
    }


def make_op():
    from problems.op.manifold import OPManifold
    N = 15
    np.random.seed(3)
    coords = np.random.rand(N + 1, 2).astype(np.float32)
    prizes = np.zeros(N + 1, dtype=np.float32)
    prizes[1:] = np.random.uniform(0.5, 2.0, N).astype(np.float32)
    dist = dist_matrix_from_coords(coords)
    return "OP", OPManifold(), {
        'coords': coords, 'prizes': prizes, 'budget': 4.0,
        'dist': dist, 'n_customers': N,
    }


def make_kp():
    from problems.kp.manifold import KPManifold
    N = 20
    np.random.seed(4)
    values = np.random.uniform(1, 10, N).astype(np.float32)
    weights = np.random.uniform(1, 5, N).astype(np.float32)
    return "KP", KPManifold(), {
        'values': values, 'weights': weights,
        'capacity': float(weights.sum() * 0.4), 'n_items': N,
    }


def make_mis():
    from problems.mis.manifold import MISManifold
    N = 20
    np.random.seed(5)
    # Random graph with ~30% edge density
    adj = np.random.random((N, N)) < 0.3
    adj = adj | adj.T  # symmetric
    np.fill_diagonal(adj, False)
    return "MIS", MISManifold(), {'adj': adj, 'n_nodes': N}


def make_mtsp():
    from problems.mtsp.manifold import MTSPManifold
    N = 15
    np.random.seed(6)
    coords = np.random.rand(N + 1, 2).astype(np.float32)
    dist = dist_matrix_from_coords(coords)
    return "mTSP", MTSPManifold(), {
        'coords': coords, 'dist': dist, 'n_customers': N, 'n_agents': 3,
    }


if __name__ == '__main__':
    problems = [make_tsp(), make_cvrp(), make_pctsp(), make_op(),
                make_kp(), make_mis(), make_mtsp()]

    for name, manifold, instance in problems:
        test_manifold(name, manifold, instance)

    print(f"\n=== ALL {len(problems)} PROBLEM MANIFOLDS PASSED ===")
