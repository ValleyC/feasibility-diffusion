"""
Clean label-free training for feasibility-preserving diffusion on TSP.

Principle: greedy 2-opt from a random start generates its own training data.
At each improvement step, we get a (tour, best_swap) pair for FREE — covering
the full quality spectrum from random to locally-optimal with zero magic numbers.

Training pipeline:
  1. Generate random TSP instance (coordinates)
  2. Run greedy 2-opt from random tour to convergence, recording each step
  3. Each step yields: (tour_at_step_k, best_swap_at_step_k)
  4. During training, sample uniformly from all recorded pairs
  5. Cross-entropy on best-swap prediction

No optimal tours needed. No external solvers. No quality-tier percentages.
The training distribution naturally matches inference because it starts
from random tours (= inference starting point) and covers all quality levels
through the 2-opt trajectory.

Usage:
    python -m training.clean_trainer --N 50 --device cuda:0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from problems.tsp.tour import (
    enumerate_2opt, apply_2opt, delta_2opt, tour_cost,
    dist_matrix_from_coords, random_tour, is_valid_tour,
)
from problems.tsp.data import generate_instance
from models.move_scorer import MoveScorer


def generate_trajectory(coords, dist, moves_list, N, max_iters=500):
    """Run greedy 2-opt from random start to convergence.

    Each step produces a training sample: (tour, best_swap_index, step/total).
    The trajectory naturally covers random → locally optimal.

    Returns:
        samples: list of (tour, label, progress) tuples
            tour: (N,) int array — tour at this step
            label: int — index of the best-improving swap (the greedy choice)
            progress: float in [0, 1] — how far along the trajectory (0=random, 1=converged)
        final_cost: float — cost of the locally-optimal tour at convergence
    """
    tour = random_tour(N)
    samples = []

    for iteration in range(max_iters):
        deltas = np.array([delta_2opt(tour, i, j, dist) for i, j in moves_list])
        best_idx = int(np.argmin(deltas))

        if deltas[best_idx] >= -1e-10:
            break  # converged to local optimum

        samples.append((tour.copy(), best_idx))
        tour = apply_2opt(tour, moves_list[best_idx][0], moves_list[best_idx][1])

    final_cost = tour_cost(tour, dist)

    # Assign progress values: 0 for first step, 1 for last
    n = len(samples)
    result = []
    for k, (t, label) in enumerate(samples):
        progress = k / max(n - 1, 1)
        result.append((t, label, progress))

    return result, final_cost


def build_sample_pool(N, n_instances, moves_list, seed=42):
    """Pre-generate a pool of training samples from multiple trajectories.

    Each instance contributes ~50-100 samples spanning random → local optimum.
    """
    print(f"Building sample pool: {n_instances} TSP-{N} trajectories...")
    t0 = time.time()

    all_coords = []
    all_dists = []
    all_samples = []  # list of (instance_idx, tour, label, progress)
    final_costs = []

    for i in range(n_instances):
        coords = generate_instance(N, seed=seed + i)
        dist = dist_matrix_from_coords(coords)
        traj, fc = generate_trajectory(coords, dist, moves_list, N)

        idx_base = len(all_coords)
        all_coords.append(coords)
        all_dists.append(dist)
        final_costs.append(fc)

        for tour, label, progress in traj:
            all_samples.append((idx_base, tour, label, progress))

    elapsed = time.time() - t0
    avg_samples = len(all_samples) / n_instances
    print(f"  {len(all_samples)} samples from {n_instances} instances "
          f"({avg_samples:.0f} steps/instance avg, {elapsed:.1f}s)")
    print(f"  Avg converged cost: {np.mean(final_costs):.4f}")

    return all_coords, all_dists, all_samples, final_costs


@torch.no_grad()
def greedy_denoise(model, coords_t, dist_np, moves_ij, N, n_steps,
                   device, n_trajectories=1):
    """Greedy denoising inference with optional multi-trajectory sampling."""
    model.eval()
    moves_list = moves_ij.cpu().numpy().tolist()

    best_tour = None
    best_cost = float('inf')

    for _ in range(n_trajectories):
        tour_np = random_tour(N)

        for step in range(n_steps):
            progress = step / max(n_steps - 1, 1)
            tour_t = torch.tensor(tour_np, dtype=torch.long, device=device).unsqueeze(0)
            t_tensor = torch.tensor([progress], device=device)

            scores = model(coords_t, tour_t, t_tensor, 1.0, moves_ij)
            best_idx = scores.argmax(dim=-1).item()
            i, j = moves_list[best_idx]

            d = delta_2opt(tour_np, i, j, dist_np)
            if d < 0:
                tour_np = apply_2opt(tour_np, i, j)

        cost = tour_cost(tour_np, dist_np)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour_np.copy()

    return best_tour, best_cost


def evaluate(model, coords_list, dist_list, ref_costs, moves_ij, N,
             n_steps, device, n_eval=None, n_trajectories=1):
    """Evaluate via greedy denoising."""
    model.eval()
    if n_eval is None:
        n_eval = len(coords_list)
    n_eval = min(n_eval, len(coords_list))

    costs, gaps = [], []
    for idx in range(n_eval):
        coords_t = torch.tensor(
            coords_list[idx], dtype=torch.float32, device=device
        ).unsqueeze(0)
        final_tour, final_cost = greedy_denoise(
            model, coords_t, dist_list[idx], moves_ij, N,
            n_steps=n_steps, device=device, n_trajectories=n_trajectories,
        )
        assert is_valid_tour(final_tour)
        costs.append(final_cost)
        gaps.append(final_cost / ref_costs[idx] - 1.0)

    return np.mean(costs), np.mean(gaps)


def train(args):
    device = torch.device(args.device)
    N = args.N

    moves_list = enumerate_2opt(N)
    M = len(moves_list)
    moves_ij = torch.tensor(moves_list, dtype=torch.long, device=device)
    print(f"TSP-{N}: {M} valid 2-opt moves")

    # Build training pool (label-free: greedy 2-opt trajectories)
    train_coords, train_dists, train_samples, _ = build_sample_pool(
        N, args.n_instances, moves_list, seed=42,
    )

    # Validation instances (separate, also self-generated)
    val_coords, val_dists, _, val_costs = build_sample_pool(
        N, args.n_val, moves_list, seed=99999,
    )

    # Model
    model = MoveScorer(
        n_node_features=5,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"MoveScorer: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs,
    )

    # Baselines
    random_costs = [
        np.mean([tour_cost(random_tour(N), val_dists[i]) for _ in range(10)])
        for i in range(len(val_dists))
    ]
    print(f"Baselines: 2-opt={np.mean(val_costs):.4f}, "
          f"random={np.mean(random_costs):.4f} "
          f"(ratio={np.mean(random_costs)/np.mean(val_costs):.2f}x)")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_gap = float('inf')
    n_samples = len(train_samples)
    n_denoise = args.n_denoise if args.n_denoise else N * 3

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0

        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch}")
        for step in pbar:
            batch_tours, batch_coords, batch_labels, batch_t = [], [], [], []

            for _ in range(args.batch_size):
                # Sample uniformly from the pre-generated pool
                s_idx = np.random.randint(n_samples)
                inst_idx, tour, label, progress = train_samples[s_idx]

                batch_tours.append(tour)
                batch_coords.append(train_coords[inst_idx])
                batch_labels.append(label)
                batch_t.append(progress)

            coords_t = torch.tensor(np.stack(batch_coords), dtype=torch.float32, device=device)
            tours_t = torch.tensor(np.stack(batch_tours), dtype=torch.long, device=device)
            labels_t = torch.tensor(batch_labels, dtype=torch.long, device=device)
            t_tensor = torch.tensor(batch_t, dtype=torch.float32, device=device)

            scores = model(coords_t, tours_t, t_tensor, 1.0, moves_ij)
            loss = F.cross_entropy(scores, labels_t)

            preds = scores.argmax(dim=-1)
            acc = (preds == labels_t).float().mean().item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_acc += acc
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{acc:.2%}")

        scheduler.step()
        avg_loss = epoch_loss / args.steps_per_epoch
        avg_acc = epoch_acc / args.steps_per_epoch
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, acc={avg_acc:.2%}")

        if epoch % args.eval_freq == 0:
            avg_cost, avg_gap = evaluate(
                model, val_coords, val_dists, val_costs,
                moves_ij, N, n_steps=n_denoise,
                device=device, n_eval=args.n_eval,
                n_trajectories=args.n_trajectories,
            )
            print(f"  Eval: cost={avg_cost:.4f}, gap={avg_gap:.2%} "
                  f"(ref={np.mean(val_costs):.4f}, "
                  f"{args.n_trajectories} traj, {n_denoise} steps)")

            if avg_gap < best_gap:
                best_gap = avg_gap
                path = os.path.join(args.ckpt_dir, f'best_tsp{N}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'gap': avg_gap,
                    'cost': avg_cost,
                    'args': vars(args),
                }, path)
                print(f"  [checkpoint] best gap={avg_gap:.2%} -> {path}")

    print(f"\nTraining complete. Best gap: {best_gap:.2%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Clean label-free training: greedy 2-opt trajectories as training data')
    parser.add_argument('--N', type=int, default=20)
    parser.add_argument('--n_instances', type=int, default=5000,
                        help='Number of random instances (each generates ~50-100 samples)')
    parser.add_argument('--n_val', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--steps_per_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_denoise', type=int, default=None,
                        help='Denoising steps at inference (default: 3*N)')
    parser.add_argument('--n_trajectories', type=int, default=1,
                        help='Multi-trajectory sampling at eval (best-of-K)')
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--n_eval', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/clean')
    args = parser.parse_args()
    train(args)
