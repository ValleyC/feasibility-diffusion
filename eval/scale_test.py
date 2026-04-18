"""
Scale generalization test: train on TSP-N, evaluate on TSP-M for M != N.

Tests whether the model (trained on one problem size) generalizes to
different sizes WITHOUT retraining — the key advantage of our
scale-agnostic architecture.

Usage:
    python -m eval.scale_test \
        --ckpt checkpoints/selfplay/round5_tsp50.pt \
        --sizes 20 50 100 200 500 \
        --n_instances 50 --n_steps 200 --device cuda:0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
import numpy as np
import torch
from tqdm import tqdm

from problems.tsp.tour import (
    enumerate_2opt, apply_2opt, delta_2opt, tour_cost,
    dist_matrix_from_coords, random_tour, is_valid_tour,
)
from problems.tsp.data import generate_instance, solve_2opt
from models.move_scorer import MoveScorer


@torch.no_grad()
def evaluate_size(model, N, n_instances, n_steps, device, n_trajectories=1, seed=777):
    """Evaluate model on TSP-N instances (model may have been trained on different N)."""
    moves_list = enumerate_2opt(N)
    M = len(moves_list)
    moves_ij = torch.tensor(moves_list, dtype=torch.long, device=device)

    model.eval()
    results = []

    pbar = tqdm(range(n_instances), desc=f"TSP-{N}", leave=False)
    for inst_id in pbar:
        coords = generate_instance(N, seed=seed + inst_id)
        dist = dist_matrix_from_coords(coords)

        # Reference: multi-start 2-opt
        ref_tour, ref_cost = solve_2opt(coords, max_restarts=min(5, N))

        # Our model: greedy denoising (best of K trajectories)
        best_cost = float('inf')
        for _ in range(n_trajectories):
            tour_np = random_tour(N)
            coords_t = torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(0)

            for step in range(n_steps):
                progress = step / max(n_steps - 1, 1)
                tour_t = torch.tensor(tour_np, dtype=torch.long, device=device).unsqueeze(0)
                t_tensor = torch.tensor([progress], device=device)

                scores = model(coords_t, tour_t, t_tensor, 1.0, moves_ij)
                best_idx = scores.argmax(dim=-1).item()
                i, j = moves_list[best_idx]

                d = delta_2opt(tour_np, i, j, dist)
                if d < 0:
                    tour_np = apply_2opt(tour_np, i, j)

            cost = tour_cost(tour_np, dist)
            assert is_valid_tour(tour_np), "Infeasible tour!"
            best_cost = min(best_cost, cost)

        gap = (best_cost / ref_cost - 1) * 100
        results.append({'ref': ref_cost, 'ours': best_cost, 'gap': gap})
        pbar.set_postfix(gap=f"{gap:+.1f}%", cost=f"{best_cost:.2f}")

    return results


def main(args):
    device = torch.device(args.device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt.get('args', {})
    train_N = ckpt_args.get('N', '?')

    model = MoveScorer(
        n_node_features=5,
        hidden_dim=ckpt_args.get('hidden_dim', 128),
        n_layers=ckpt_args.get('n_layers', 6),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Trained on TSP-{train_N}, {sum(p.numel() for p in model.parameters()):,} params")

    # Test on each size
    print(f"\n{'='*65}")
    print(f"{'N':>6} | {'Ref (2-opt)':>12} | {'Ours':>12} | {'Gap':>8} | {'Time':>8}")
    print(f"{'='*65}")

    for N in args.sizes:
        n_steps = max(N * 3, args.n_steps)
        t0 = time.time()
        results = evaluate_size(
            model, N, args.n_instances, n_steps, device,
            n_trajectories=args.n_trajectories,
        )
        elapsed = time.time() - t0

        avg_ref = np.mean([r['ref'] for r in results])
        avg_ours = np.mean([r['ours'] for r in results])
        avg_gap = np.mean([r['gap'] for r in results])

        print(f"{N:>6} | {avg_ref:>12.4f} | {avg_ours:>12.4f} | {avg_gap:>+7.2f}% | {elapsed:>7.1f}s")

    print(f"{'='*65}")
    print(f"All tours verified FEASIBLE at every size.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--sizes', nargs='+', type=int, default=[20, 50, 100, 200])
    parser.add_argument('--n_instances', type=int, default=50)
    parser.add_argument('--n_steps', type=int, default=150)
    parser.add_argument('--n_trajectories', type=int, default=1)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    main(args)
