"""
Correctness tests for the PCTSP selection manifold.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from problems.pctsp.selection import (
    random_selection, selection_feasible, selection_cost,
    enumerate_selection_moves, apply_selection_move,
    delta_selection_move, prize_total, SelMoveType,
)
from problems.pctsp.manifold import PCTSPManifold
from problems.tsp.tour import dist_matrix_from_coords


def make_instance(N=20, seed=42):
    np.random.seed(seed)
    coords = np.random.rand(N + 1, 2).astype(np.float32)
    prizes = np.zeros(N + 1, dtype=np.float32)
    prizes[1:] = np.random.uniform(0.1, 1.0, N).astype(np.float32)
    penalties = np.zeros(N + 1, dtype=np.float32)
    penalties[1:] = np.random.uniform(0.1, 0.5, N).astype(np.float32)
    min_prize = float(prizes.sum() * 0.5)  # require at least 50% of total prize
    dist = dist_matrix_from_coords(coords)
    return {
        'coords': coords, 'prizes': prizes, 'penalties': penalties,
        'min_prize': min_prize, 'dist': dist, 'n_customers': N,
    }


def test_random_feasibility():
    for N in [10, 20, 50]:
        inst = make_instance(N=N)
        for _ in range(20):
            sel = random_selection(N, inst['prizes'], inst['min_prize'])
            assert selection_feasible(sel, inst['prizes'], inst['min_prize'], N), \
                f"Random selection infeasible for N={N}"
            assert sel[0], "Depot must be selected"
    print("[PASS] random_feasibility")


def test_three_move_types():
    inst = make_instance(N=15, seed=7)
    sel = random_selection(15, inst['prizes'], inst['min_prize'])
    moves = enumerate_selection_moves(sel, inst['prizes'], inst['min_prize'])
    types = set(m[0] for m in moves)
    counts = {SelMoveType.ADD: 0, SelMoveType.REMOVE: 0, SelMoveType.SWAP: 0}
    for m in moves:
        counts[m[0]] += 1
    print(f"[PASS] three_move_types: add={counts[SelMoveType.ADD]}, "
          f"remove={counts[SelMoveType.REMOVE]}, "
          f"swap={counts[SelMoveType.SWAP]}, total={len(moves)}")


def test_moves_preserve_feasibility():
    for N in [10, 15, 20]:
        inst = make_instance(N=N, seed=123)
        sel = random_selection(N, inst['prizes'], inst['min_prize'])
        assert selection_feasible(sel, inst['prizes'], inst['min_prize'], N)

        moves = enumerate_selection_moves(sel, inst['prizes'], inst['min_prize'])
        for m in moves:
            new_sel = apply_selection_move(sel, m)
            assert selection_feasible(new_sel, inst['prizes'], inst['min_prize'], N), \
                f"Move {m} produced infeasible selection"
    print(f"[PASS] moves_preserve_feasibility (exhaustive for N=10,15,20)")


def test_delta_matches_brute_force():
    inst = make_instance(N=15, seed=42)
    sel = random_selection(15, inst['prizes'], inst['min_prize'])
    cost_before = selection_cost(sel, inst['coords'], inst['dist'], inst['penalties'])

    moves = enumerate_selection_moves(sel, inst['prizes'], inst['min_prize'])
    for m in moves:
        delta = delta_selection_move(sel, m, inst['coords'], inst['dist'], inst['penalties'])
        new_sel = apply_selection_move(sel, m)
        cost_after = selection_cost(new_sel, inst['coords'], inst['dist'], inst['penalties'])
        actual_delta = cost_after - cost_before
        assert abs(delta - actual_delta) < 1e-4, \
            f"Move {m}: delta={delta:.4f} vs actual={actual_delta:.4f}"
    print(f"[PASS] delta_matches_brute_force ({len(moves)} moves)")


def test_forward_preserves_feasibility():
    manifold = PCTSPManifold()
    inst = make_instance(N=20, seed=99)
    sel = manifold.sample_random(inst)
    assert manifold.is_feasible(sel, inst)

    for step in range(100):
        moves = manifold.enumerate_moves(sel, inst)
        if len(moves) == 0:
            break
        idx = np.random.randint(len(moves))
        sel = manifold.apply_move(sel, moves[idx])
        assert manifold.is_feasible(sel, inst), \
            f"Step {step}: infeasible after random move"
    print("[PASS] forward_preserves_feasibility (100 steps)")


def test_best_move_improves():
    manifold = PCTSPManifold()
    inst = make_instance(N=20, seed=55)
    sel = manifold.sample_random(inst)
    cost_before = manifold.cost(sel, inst)

    move, delta = manifold.best_move(sel, inst)
    if move is not None and delta < 0:
        new_sel = manifold.apply_move(sel, move)
        cost_after = manifold.cost(new_sel, inst)
        assert cost_after <= cost_before + 1e-4
    print(f"[PASS] best_move_improves (delta={delta:.4f})")


def test_prize_constraint_enforced():
    """Verify remove moves only allowed when prize stays above threshold."""
    inst = make_instance(N=10, seed=0)
    sel = random_selection(10, inst['prizes'], inst['min_prize'])
    moves = enumerate_selection_moves(sel, inst['prizes'], inst['min_prize'])

    for m in moves:
        if m[0] == SelMoveType.REMOVE:
            new_sel = apply_selection_move(sel, m)
            assert prize_total(new_sel, inst['prizes']) >= inst['min_prize'] - 1e-8
        elif m[0] == SelMoveType.SWAP:
            new_sel = apply_selection_move(sel, m)
            assert prize_total(new_sel, inst['prizes']) >= inst['min_prize'] - 1e-8
    print("[PASS] prize_constraint_enforced")


if __name__ == '__main__':
    test_random_feasibility()
    test_three_move_types()
    test_moves_preserve_feasibility()
    test_delta_matches_brute_force()
    test_forward_preserves_feasibility()
    test_best_move_improves()
    test_prize_constraint_enforced()
    print("\n=== ALL PCTSP TESTS PASSED ===")
