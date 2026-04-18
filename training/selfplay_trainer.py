"""
Self-play training for feasibility-preserving diffusion on TSP.

Principle: the model bootstraps toward optimality via iterative self-improvement.

  Round 1: Train on greedy 2-opt trajectories (ceiling = 2-opt quality)
  Round 2: Run best-of-K inference → find tours better than 2-opt → retrain
  Round 3: Run best-of-K on improved model → even better tours → retrain
  ...converges toward optimal

No external solver. Every state feasible. Labels from own inference.
This is the AlphaZero self-play principle applied to CO.

Usage:
    python -m training.selfplay_trainer --N 50 --device cuda:0 --n_rounds 5
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
import copy
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from problems.tsp.tour import (
    enumerate_2opt, apply_2opt, delta_2opt, tour_cost,
    dist_matrix_from_coords, random_tour, is_valid_tour,
)
from problems.tsp.data import generate_instance, solve_2opt
from models.move_scorer import MoveScorer


# ── Data generation ──────────────────────────────────────────

def generate_instances(N, n_instances, seed=42):
    """Generate random TSP instances (coordinates + distance matrices)."""
    coords_list, dist_list = [], []
    for i in range(n_instances):
        coords = generate_instance(N, seed=seed + i)
        dist = dist_matrix_from_coords(coords)
        coords_list.append(coords)
        dist_list.append(dist)
    return coords_list, dist_list


def generate_2opt_targets(coords_list, dist_list, moves_list, N):
    """Generate initial targets via greedy 2-opt (Round 0 bootstrap)."""
    print(f"Generating initial 2-opt targets for {len(coords_list)} instances...")
    targets = []
    for coords, dist in zip(coords_list, dist_list):
        tour, cost = solve_2opt(coords, max_restarts=3)
        targets.append((tour, cost))
    avg = np.mean([c for _, c in targets])
    print(f"  Initial 2-opt avg cost: {avg:.4f}")
    return targets


def build_trajectory_samples(coords_list, dist_list, targets, moves_list, N):
    """Build training samples from greedy 2-opt trajectories toward current targets.

    For each instance, run greedy 2-opt from random start, recording
    (tour, best_swap) at each step. The target cost is used for reference only.
    """
    all_samples = []
    for idx, (coords, dist) in enumerate(zip(coords_list, dist_list)):
        tour = random_tour(N)
        for iteration in range(N * 10):
            deltas = np.array([delta_2opt(tour, i, j, dist) for i, j in moves_list])
            best_idx = int(np.argmin(deltas))
            if deltas[best_idx] >= -1e-10:
                break
            all_samples.append((idx, tour.copy(), best_idx))
            tour = apply_2opt(tour, moves_list[best_idx][0], moves_list[best_idx][1])
    return all_samples


def build_target_guided_samples(coords_list, dist_list, targets, moves_list, N,
                                n_scramble_max=None):
    """Build training samples guided by the current best targets.

    For each instance:
    1. Take the current best tour (target)
    2. Scramble it by t random 2-opt swaps (t ~ U[1, T])
    3. At the scrambled state, the label = best-improving swap

    This teaches the model to navigate TOWARD the target from any quality level.
    Combines the strengths of trajectory-based and target-based training.
    """
    if n_scramble_max is None:
        n_scramble_max = N * 3
    all_samples = []
    for idx, (coords, dist) in enumerate(zip(coords_list, dist_list)):
        target_tour, target_cost = targets[idx]

        # Generate samples at multiple noise levels from the target
        for _ in range(5):  # 5 scramble levels per instance
            t = np.random.randint(1, n_scramble_max + 1)
            tour = target_tour.copy()
            for _ in range(t):
                swap_idx = np.random.randint(len(moves_list))
                tour = apply_2opt(tour, moves_list[swap_idx][0], moves_list[swap_idx][1])

            # Label: best-improving swap at this scrambled state
            deltas = np.array([delta_2opt(tour, i, j, dist) for i, j in moves_list])
            best_idx = int(np.argmin(deltas))
            if deltas[best_idx] < -1e-10:
                all_samples.append((idx, tour.copy(), best_idx))

        # Also add samples from random starts (coverage of full quality range)
        tour = random_tour(N)
        for iteration in range(N * 2):
            deltas = np.array([delta_2opt(tour, i, j, dist) for i, j in moves_list])
            best_idx = int(np.argmin(deltas))
            if deltas[best_idx] >= -1e-10:
                break
            all_samples.append((idx, tour.copy(), best_idx))
            tour = apply_2opt(tour, moves_list[best_idx][0], moves_list[best_idx][1])

    return all_samples


# ── Inference ────────────────────────────────────────────────

@torch.no_grad()
def denoise_bestofk(model, coords, dist, moves_ij, N, n_steps,
                    device, K=8):
    """Run K denoising trajectories with stochastic sampling, return best tour."""
    model.eval()
    moves_list = moves_ij.cpu().numpy().tolist()
    coords_t = torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(0)

    best_tour, best_cost = None, float('inf')

    for _ in range(K):
        tour_np = random_tour(N)

        for step in range(n_steps):
            progress = step / max(n_steps - 1, 1)
            tour_t = torch.tensor(tour_np, dtype=torch.long, device=device).unsqueeze(0)
            t_tensor = torch.tensor([progress], device=device)
            scores = model(coords_t, tour_t, t_tensor, 1.0, moves_ij)

            # Stochastic sampling (temperature-controlled) for exploration
            if step < n_steps * 0.7:
                # Early steps: explore (sample from softmax)
                probs = F.softmax(scores / 0.5, dim=-1)
                action = torch.multinomial(probs.squeeze(0), 1).item()
            else:
                # Late steps: exploit (greedy)
                action = scores.argmax(dim=-1).item()

            i, j = moves_list[action]
            d = delta_2opt(tour_np, i, j, dist)
            if d < 0:
                tour_np = apply_2opt(tour_np, i, j)

        cost = tour_cost(tour_np, dist)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour_np.copy()

    return best_tour, best_cost


# ── Training ─────────────────────────────────────────────────

def train_one_round(model, optimizer, scheduler, samples, coords_list,
                    moves_ij, N, n_epochs, steps_per_epoch, batch_size,
                    device):
    """Train the model for one self-play round."""
    n_samples = len(samples)

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss, epoch_acc = 0.0, 0.0

        pbar = tqdm(range(steps_per_epoch), desc=f"  Epoch {epoch}", leave=False)
        for step in pbar:
            batch_tours, batch_coords, batch_labels, batch_t = [], [], [], []

            for _ in range(batch_size):
                s_idx = np.random.randint(n_samples)
                inst_idx, tour, label = samples[s_idx]
                # Progress estimate based on tour cost relative to instance
                batch_tours.append(tour)
                batch_coords.append(coords_list[inst_idx])
                batch_labels.append(label)
                batch_t.append(np.random.random())  # random progress signal

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
        print(f"  Epoch {epoch}: loss={epoch_loss/steps_per_epoch:.4f}, "
              f"acc={epoch_acc/steps_per_epoch:.2%}")


def selfplay_improve(model, coords_list, dist_list, targets, moves_ij,
                     N, n_steps, device, K=8, n_improve=None):
    """Run best-of-K inference on all instances, upgrade targets where improved.

    Returns:
        new_targets: updated list of (tour, cost)
        n_improved: how many instances got a better tour
    """
    if n_improve is None:
        n_improve = len(coords_list)
    n_improve = min(n_improve, len(coords_list))

    new_targets = list(targets)  # shallow copy
    n_improved = 0

    for idx in tqdm(range(n_improve), desc="  Self-play inference"):
        old_tour, old_cost = targets[idx]
        new_tour, new_cost = denoise_bestofk(
            model, coords_list[idx], dist_list[idx],
            moves_ij, N, n_steps, device, K=K,
        )
        if new_cost < old_cost - 1e-8:
            new_targets[idx] = (new_tour, new_cost)
            n_improved += 1

    old_avg = np.mean([c for _, c in targets[:n_improve]])
    new_avg = np.mean([c for _, c in new_targets[:n_improve]])
    print(f"  Self-play: {n_improved}/{n_improve} improved, "
          f"avg cost {old_avg:.4f} -> {new_avg:.4f} "
          f"(delta: {(new_avg/old_avg - 1)*100:.2f}%)")

    return new_targets, n_improved


# ── Main ─────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device)
    N = args.N

    moves_list = enumerate_2opt(N)
    M = len(moves_list)
    moves_ij = torch.tensor(moves_list, dtype=torch.long, device=device)
    print(f"TSP-{N}: {M} valid 2-opt moves")

    # Generate instances
    coords_train, dist_train = generate_instances(N, args.n_train, seed=42)
    coords_val, dist_val = generate_instances(N, args.n_val, seed=99999)

    # Initial targets via greedy 2-opt
    targets_train = generate_2opt_targets(coords_train, dist_train, moves_list, N)
    targets_val = generate_2opt_targets(coords_val, dist_val, moves_list, N)

    # Model
    model = MoveScorer(
        n_node_features=5,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(device)
    print(f"MoveScorer: {sum(p.numel() for p in model.parameters()):,} parameters")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    n_steps_infer = args.n_steps if args.n_steps else N * 3

    # ── Self-play rounds ──
    for round_id in range(1, args.n_rounds + 1):
        print(f"\n{'='*60}")
        print(f"ROUND {round_id}/{args.n_rounds}")
        print(f"{'='*60}")

        # Build training samples from current targets
        print("Building training samples...")
        if round_id == 1:
            # First round: use trajectory-based samples (from random starts)
            samples = build_trajectory_samples(
                coords_train, dist_train, targets_train, moves_list, N,
            )
        else:
            # Subsequent rounds: target-guided samples (scramble from best tours)
            samples = build_target_guided_samples(
                coords_train, dist_train, targets_train, moves_list, N,
            )
        print(f"  {len(samples)} training samples")

        # Fresh optimizer each round (prevent stale momentum)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs_per_round,
        )

        # Train
        train_one_round(
            model, optimizer, scheduler, samples, coords_train,
            moves_ij, N, args.epochs_per_round, args.steps_per_epoch,
            args.batch_size, device,
        )

        # Evaluate on validation set
        print("Evaluating on validation...")
        val_costs = []
        for idx in range(min(args.n_eval, len(coords_val))):
            _, cost = denoise_bestofk(
                model, coords_val[idx], dist_val[idx],
                moves_ij, N, n_steps_infer, device, K=args.K,
            )
            val_costs.append(cost)
        val_avg = np.mean(val_costs)
        ref_avg = np.mean([c for _, c in targets_val[:len(val_costs)]])
        gap = (val_avg / ref_avg - 1) * 100
        print(f"  Val: cost={val_avg:.4f}, gap={gap:.2f}% vs 2-opt ref={ref_avg:.4f}")

        # Save checkpoint
        ckpt_path = os.path.join(args.ckpt_dir, f'round{round_id}_tsp{N}.pt')
        torch.save({
            'round': round_id,
            'model_state_dict': model.state_dict(),
            'val_cost': val_avg,
            'val_gap': gap,
            'args': vars(args),
        }, ckpt_path)
        print(f"  Saved: {ckpt_path}")

        # Self-play: improve training targets
        if round_id < args.n_rounds:
            print("Self-play improvement...")
            targets_train, n_improved = selfplay_improve(
                model, coords_train, dist_train, targets_train,
                moves_ij, N, n_steps_infer, device,
                K=args.K, n_improve=args.n_improve,
            )
            if n_improved == 0:
                print("  No improvements found. Stopping early.")
                break

    # Final summary
    final_avg = np.mean([c for _, c in targets_train])
    init_ref = np.mean([c for _, c in generate_2opt_targets(
        coords_train[:10], dist_train[:10], moves_list, N)])
    print(f"\nSelf-play complete.")
    print(f"  Final training target avg: {final_avg:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Self-play feasibility-preserving diffusion')
    parser.add_argument('--N', type=int, default=20)
    parser.add_argument('--n_train', type=int, default=2000)
    parser.add_argument('--n_val', type=int, default=50)
    parser.add_argument('--n_rounds', type=int, default=5, help='Number of self-play rounds')
    parser.add_argument('--epochs_per_round', type=int, default=15)
    parser.add_argument('--steps_per_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--K', type=int, default=8, help='Best-of-K trajectories for self-play')
    parser.add_argument('--n_steps', type=int, default=None, help='Denoising steps (default: 3*N)')
    parser.add_argument('--n_eval', type=int, default=20)
    parser.add_argument('--n_improve', type=int, default=None,
                        help='Instances to run self-play on per round (default: all)')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/selfplay')
    args = parser.parse_args()
    train(args)
