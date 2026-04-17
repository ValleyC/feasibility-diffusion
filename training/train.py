"""
Supervised training for feasibility-preserving diffusion on TSP.

Training loop:
  1. Generate TSP instances + near-optimal tours (offline, cached)
  2. For each step:
     a. Pick a random instance
     b. Scramble optimal tour by t random 2-opt swaps (t ~ U[1, T_max])
     c. Compute best-improving swap on scrambled tour (= label)
     d. Model scores all N(N-3)/2 moves
     e. Cross-entropy loss: predict the best swap

Inference (greedy denoising):
  1. Start from a random valid tour
  2. For T steps: apply the model's highest-scored 2-opt swap
  3. Report final tour cost

Usage:
    cd feasible-diffusion
    python -m training.train --N 20 --n_train 1000 --n_epochs 30 --device cuda:0
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
from problems.tsp.data import generate_dataset
from problems.tsp.edisco_data import load_edisco_tsp
from models.move_scorer import MoveScorer


def prepare_dataset(N, n_instances, seed=42, solver_restarts=5,
                    data_path=None, max_instances=None):
    """Load or generate instances + near-optimal tours.

    If data_path is given, loads EDISCO-format data from file.
    Otherwise, generates random instances and solves with 2-opt.
    """
    if data_path is not None:
        return load_edisco_tsp(data_path, max_instances=max_instances)

    print(f"Generating {n_instances} TSP-{N} instances with 2-opt solutions...")
    t0 = time.time()
    coords_list, dist_list, tour_list, cost_list = generate_dataset(
        N, n_instances, seed=seed, solver_restarts=solver_restarts,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s. Avg optimal cost: {np.mean(cost_list):.4f}")
    return coords_list, dist_list, tour_list, cost_list


def make_training_sample(coords, dist, opt_tour, t, moves_list, N):
    """Create one training sample: scramble, compute label.

    Returns:
        scrambled_tour: (N,) int
        t: int (noise level)
        label: int (index of best-improving move)
        best_delta: float (cost change of best move)
    """
    # Scramble
    tour = opt_tour.copy()
    for _ in range(t):
        idx = np.random.randint(len(moves_list))
        i, j = moves_list[idx]
        tour = apply_2opt(tour, i, j)

    # Find best improving move (supervised label)
    deltas = np.array([delta_2opt(tour, i, j, dist) for i, j in moves_list])
    label = int(np.argmin(deltas))

    return tour, t, label, deltas[label]


def make_training_sample_mixed(coords, dist, moves_list, N,
                               opt_tour=None, quality_mix=(0.3, 0.3, 0.4)):
    """Create training sample from mixed-quality tours (no distribution mismatch).

    Samples tours at different quality levels so the model learns to improve
    tours of ANY quality, not just near-optimal ones:
      - Random tours (worst quality, matches inference starting point)
      - Partially improved tours (mid quality, via a few greedy 2-opt steps)
      - Near-optimal tours (best quality, from training data or heavy 2-opt)

    Args:
        coords: (N, 2) node coordinates
        dist: (N, N) distance matrix
        moves_list: list of (i, j) valid 2-opt moves
        N: number of nodes
        opt_tour: optional near-optimal tour (used for "good" quality tier)
        quality_mix: (p_random, p_partial, p_good) probabilities for each tier

    Returns:
        tour: (N,) int — tour at some quality level
        quality: float — normalized quality indicator in [0, 1]
        label: int — index of best-improving move
        best_delta: float
    """
    p_random, p_partial, p_good = quality_mix
    roll = np.random.random()

    if roll < p_random:
        # Tier 1: fully random tour (matches inference starting point)
        tour = random_tour(N)
        quality = 0.0
    elif roll < p_random + p_partial:
        # Tier 2: random tour improved by k greedy 2-opt steps
        tour = random_tour(N)
        k = np.random.randint(1, N * 2)  # 1 to 2N improvement steps
        for _ in range(k):
            deltas_k = np.array([delta_2opt(tour, i, j, dist) for i, j in moves_list])
            best_k = int(np.argmin(deltas_k))
            if deltas_k[best_k] >= -1e-10:
                break  # locally optimal
            tour = apply_2opt(tour, moves_list[best_k][0], moves_list[best_k][1])
        quality = 0.5
    else:
        # Tier 3: near-optimal (from data, or scrambled slightly from optimal)
        if opt_tour is not None:
            tour = opt_tour.copy()
            # Light scramble (0-10 swaps) so model also sees near-optimal states
            n_scramble = np.random.randint(0, 11)
            for _ in range(n_scramble):
                idx = np.random.randint(len(moves_list))
                tour = apply_2opt(tour, moves_list[idx][0], moves_list[idx][1])
        else:
            # Fallback: run greedy 2-opt to convergence
            tour = random_tour(N)
            for _ in range(N * 10):
                deltas_k = np.array([delta_2opt(tour, i, j, dist) for i, j in moves_list])
                best_k = int(np.argmin(deltas_k))
                if deltas_k[best_k] >= -1e-10:
                    break
                tour = apply_2opt(tour, moves_list[best_k][0], moves_list[best_k][1])
        quality = 1.0

    # Label: best improving move on this tour
    deltas = np.array([delta_2opt(tour, i, j, dist) for i, j in moves_list])
    label = int(np.argmin(deltas))

    return tour, quality, label, deltas[label]


@torch.no_grad()
def greedy_denoise(model, coords_t, dist_np, moves_ij, N, n_steps, t_max,
                   device, start_tour=None):
    """Greedy denoising: repeatedly apply highest-scored move.

    Args:
        model: MoveScorer
        coords_t: (1, N, 2) tensor
        dist_np: (N, N) numpy distance matrix
        moves_ij: (M, 2) tensor of move indices
        N: number of nodes
        n_steps: denoising steps
        t_max: max noise level (for time embedding)
        device: torch device
        start_tour: optional starting tour; if None, random

    Returns:
        final_tour: (N,) numpy array
        cost_trajectory: list of costs at each step
    """
    model.eval()
    moves_list = moves_ij.cpu().numpy().tolist()

    if start_tour is None:
        tour_np = random_tour(N)
    else:
        tour_np = start_tour.copy()

    cost_trajectory = [tour_cost(tour_np, dist_np)]

    for step in range(n_steps):
        # Quality signal: starts at 0 (random), increases toward 1 (optimal)
        # as denoising progresses
        quality = step / max(n_steps - 1, 1)

        tour_t = torch.tensor(tour_np, dtype=torch.long, device=device).unsqueeze(0)
        t_tensor = torch.tensor([quality], device=device)

        scores = model(coords_t, tour_t, t_tensor, 1.0, moves_ij)  # (1, M)
        best_idx = scores.argmax(dim=-1).item()
        i, j = moves_list[best_idx]

        # Only apply if it improves (greedy with guard)
        d = delta_2opt(tour_np, i, j, dist_np)
        if d < 0:
            tour_np = apply_2opt(tour_np, i, j)

        cost_trajectory.append(tour_cost(tour_np, dist_np))

    return tour_np, cost_trajectory


def evaluate(model, coords_list, dist_list, opt_costs, moves_ij, N,
             n_denoise_steps, t_max, device, n_eval=None):
    """Evaluate model on test instances via greedy denoising.

    Returns:
        avg_cost: average final tour cost
        avg_gap: average gap to optimal (cost / opt_cost - 1)
    """
    model.eval()
    if n_eval is None:
        n_eval = len(coords_list)
    n_eval = min(n_eval, len(coords_list))

    costs, gaps = [], []
    for idx in range(n_eval):
        coords_t = torch.tensor(
            coords_list[idx], dtype=torch.float32, device=device
        ).unsqueeze(0)
        dist_np = dist_list[idx]
        opt_cost = opt_costs[idx]

        final_tour, _ = greedy_denoise(
            model, coords_t, dist_np, moves_ij, N,
            n_steps=n_denoise_steps, t_max=t_max, device=device,
        )
        final_cost = tour_cost(final_tour, dist_np)
        assert is_valid_tour(final_tour), "Denoised tour is infeasible!"

        costs.append(final_cost)
        gaps.append(final_cost / opt_cost - 1.0)

    return np.mean(costs), np.mean(gaps)


def train(args):
    device = torch.device(args.device)
    N = args.N
    t_max = args.t_max if args.t_max else N * 2  # default: 2N scrambling steps

    # Precompute move list
    moves_list = enumerate_2opt(N)
    M = len(moves_list)
    moves_ij = torch.tensor(moves_list, dtype=torch.long, device=device)
    print(f"TSP-{N}: {M} valid 2-opt moves per tour")

    # Load or generate dataset
    n_train = args.n_train
    n_val = args.n_val
    coords_train, dist_train, tour_train, cost_train = prepare_dataset(
        N, n_train, seed=42, solver_restarts=max(3, N // 5),
        data_path=args.train_data, max_instances=n_train,
    )
    coords_val, dist_val, tour_val, cost_val = prepare_dataset(
        N, n_val, seed=99999, solver_restarts=max(3, N // 5),
        data_path=args.val_data, max_instances=n_val,
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

    # Baselines (use actual number of loaded val instances, not n_val arg)
    n_val_actual = len(dist_val)
    random_costs = []
    for idx in range(n_val_actual):
        rc = np.mean([tour_cost(random_tour(N), dist_val[idx]) for _ in range(10)])
        random_costs.append(rc)
    print(f"Baselines: optimal={np.mean(cost_val):.4f}, "
          f"random={np.mean(random_costs):.4f} "
          f"(ratio={np.mean(random_costs)/np.mean(cost_val):.2f}x) "
          f"[{n_val_actual} val instances]")

    # Checkpointing
    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_gap = float('inf')

    steps_per_epoch = args.steps_per_epoch
    batch_size = args.batch_size

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_acc = 0.0

        pbar = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch}")
        for step in pbar:
            # Build batch from mixed-quality tours
            batch_tours = []
            batch_coords = []
            batch_labels = []
            batch_t = []

            for _ in range(batch_size):
                idx = np.random.randint(len(coords_train))
                tour, quality, label, _ = make_training_sample_mixed(
                    coords_train[idx], dist_train[idx], moves_list, N,
                    opt_tour=tour_train[idx],
                )
                batch_tours.append(tour)
                batch_coords.append(coords_train[idx])
                batch_labels.append(label)
                batch_t.append(quality)  # quality as time signal: 0=random, 1=optimal

            coords_t = torch.tensor(
                np.stack(batch_coords), dtype=torch.float32, device=device
            )
            tours_t = torch.tensor(
                np.stack(batch_tours), dtype=torch.long, device=device
            )
            labels_t = torch.tensor(batch_labels, dtype=torch.long, device=device)
            t_tensor = torch.tensor(batch_t, dtype=torch.float32, device=device)

            # Forward (t_max=1.0 since quality is normalized to [0,1])
            scores = model(coords_t, tours_t, t_tensor, 1.0, moves_ij)  # (B, M)

            # Cross-entropy loss
            loss = F.cross_entropy(scores, labels_t)

            # Accuracy
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
        avg_loss = epoch_loss / steps_per_epoch
        avg_acc = epoch_acc / steps_per_epoch
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, acc={avg_acc:.2%}")

        # Evaluate
        if epoch % args.eval_freq == 0:
            avg_cost, avg_gap = evaluate(
                model, coords_val, dist_val, cost_val,
                moves_ij, N, n_denoise_steps=t_max, t_max=t_max,
                device=device, n_eval=args.n_eval,
            )
            print(f"  Eval: cost={avg_cost:.4f}, gap={avg_gap:.2%} "
                  f"(optimal={np.mean(cost_val):.4f})")

            if avg_gap < best_gap:
                best_gap = avg_gap
                path = os.path.join(args.ckpt_dir, f'best_tsp{N}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'gap': avg_gap,
                    'cost': avg_cost,
                }, path)
                print(f"  [checkpoint] best gap={avg_gap:.2%} -> {path}")

    print(f"\nTraining complete. Best gap: {best_gap:.2%}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--N', type=int, default=20, help='Number of TSP nodes')
    parser.add_argument('--n_train', type=int, default=1000, help='Training instances')
    parser.add_argument('--n_val', type=int, default=100, help='Validation instances')
    parser.add_argument('--train_data', type=str, default=None,
                        help='Path to EDISCO-format training data (skip generation if given)')
    parser.add_argument('--val_data', type=str, default=None,
                        help='Path to EDISCO-format validation data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--steps_per_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--t_max', type=int, default=None,
                        help='Max scrambling steps (default: 2*N)')
    parser.add_argument('--eval_freq', type=int, default=3)
    parser.add_argument('--n_eval', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    args = parser.parse_args()
    train(args)
