"""
FMD self-play trainer for ANY CO problem.

Same self-play principle but with proper diffusion formulation:
  Round 1: Train on quality-diverse solutions with Boltzmann soft targets
  Round 2+: Best-of-K stochastic inference → upgrade targets → retrain

Stochastic sampling provides diversity; best-of-K improves quality each round.

Usage:
    python -m training.generic_selfplay --problem cvrp --N 50 --device cuda:0
    python -m training.generic_selfplay --problem tsp --N 50 --device cuda:0
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

from models.problem_configs import PROBLEM_CONFIGS
from training.generic_trainer import (
    GenericMoveScorer, GNNLayer, build_sample_pool,
    prepare_batch_item, greedy_denoise, stochastic_denoise,
    best_of_k_denoise, boltzmann_target, calibrate_tau,
)


def selfplay_improve(model, config, manifold, instances, targets,
                     max_moves, n_steps, device, K=8, n_improve=None,
                     temperature=0.5):
    """Best-of-K stochastic inference, upgrade targets where improved."""
    if n_improve is None:
        n_improve = len(instances)
    n_improve = min(n_improve, len(instances))

    new_targets = list(targets)
    n_improved = 0

    pbar = tqdm(range(n_improve), desc="  Self-play", leave=False)
    for idx in pbar:
        old_cost = targets[idx]

        _, best_cost = best_of_k_denoise(
            model, config, manifold, instances[idx],
            max_moves, n_steps, device, K=K, temperature=temperature,
        )

        if best_cost < old_cost - 1e-8:
            new_targets[idx] = best_cost
            n_improved += 1

        pbar.set_postfix(improved=f"{n_improved}/{idx+1}")

    old_avg = np.mean(targets[:n_improve])
    new_avg = np.mean(new_targets[:n_improve])
    print(f"  Self-play: {n_improved}/{n_improve} improved, "
          f"avg {old_avg:.4f} -> {new_avg:.4f} ({(new_avg/old_avg-1)*100:+.2f}%)")
    return new_targets, n_improved


def train(args):
    device = torch.device(args.device)

    ConfigClass = PROBLEM_CONFIGS[args.problem]

    sub_solver_fn = None
    if hasattr(args, 'sub_tsp') and args.sub_tsp and args.sub_tsp != '2opt':
        from solvers.batched_subtsp import BatchedSubTSPSolver
        reviser_path = getattr(args, 'reviser_path', None)
        solver = BatchedSubTSPSolver(
            reviser_path=reviser_path, device=device, method=args.sub_tsp
        )
        sub_solver_fn = solver.solve_single
        print(f"Sub-TSP solver: {args.sub_tsp}")

    if sub_solver_fn is not None and hasattr(ConfigClass, '__init__'):
        config = ConfigClass(sub_solver_fn=sub_solver_fn)
    else:
        config = ConfigClass()
    manifold = config.create_manifold()
    print(f"Problem: {config.name}, N={args.N}, K={args.K} trajectories")

    # Generate instances
    print(f"Generating {args.n_instances} instances...")
    instances = [config.create_instance(args.N, seed=42 + i)
                 for i in tqdm(range(args.n_instances), desc="  Instances")]
    val_instances = [config.create_instance(args.N, seed=99999 + i)
                     for i in range(args.n_val)]

    os.makedirs(args.ckpt_dir, exist_ok=True)
    n_denoise = args.n_denoise if args.n_denoise else max(args.N * 3, 50)
    best_val_gap = float('inf')

    # Auto-calibrate temperature
    print("Calibrating temperature range...")
    tau_min, tau_max = calibrate_tau(manifold, instances, args.max_moves)
    print(f"  tau_min={tau_min:.4f}, tau_max={tau_max:.4f}")

    # Build initial targets BEFORE creating model on GPU
    print("Building initial targets...")
    _, train_targets = build_sample_pool(
        config, manifold, instances, args.max_moves,
        n_quality_levels=args.n_quality_levels,
    )
    _, val_targets = build_sample_pool(
        config, manifold, val_instances, args.max_moves,
        n_quality_levels=args.n_quality_levels,
    )
    print(f"Initial: train={np.mean(train_targets):.4f}, val={np.mean(val_targets):.4f}")

    # Model — create AFTER pool generation to avoid CUDA fork issues
    model = GenericMoveScorer(
        n_node_features=config.n_node_features,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(device)
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    for round_id in range(1, args.n_rounds + 1):
        print(f"\n{'='*60}")
        print(f"ROUND {round_id}/{args.n_rounds}")
        print(f"{'='*60}")

        # Build training samples
        print("Building training samples...")
        t0 = time.time()
        pool, _ = build_sample_pool(
            config, manifold, instances, args.max_moves,
            n_quality_levels=args.n_quality_levels,
            n_workers=1 if round_id > 1 else None,
        )
        print(f"  {len(pool)} samples in {time.time()-t0:.1f}s")

        if len(pool) == 0:
            print("  No training samples! Skipping round.")
            continue

        n_pool = len(pool)

        # Fresh optimizer per round
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs_per_round
        )

        # Train
        for epoch in range(1, args.epochs_per_round + 1):
            model.train()
            epoch_loss, epoch_acc = 0.0, 0.0

            pbar = tqdm(range(args.steps_per_epoch), desc=f"  Epoch {epoch}", leave=False)
            for step in pbar:
                batch_nf, batch_ei, batch_mn, batch_mm = [], [], [], []
                batch_targets, batch_t = [], []
                max_edges = 0

                items = []
                for _ in range(args.batch_size):
                    s_idx = np.random.randint(n_pool)
                    idx, sol, deltas, t = pool[s_idx]

                    tau_t = tau_min + t * (tau_max - tau_min)
                    target = boltzmann_target(deltas, tau_t)

                    moves = manifold.enumerate_moves(sol, instances[idx])
                    if len(moves) != len(deltas):
                        continue

                    nf, ei, mn, mm, _ = prepare_batch_item(
                        config, sol, instances[idx], moves, 0, t, args.max_moves
                    )
                    items.append((nf, ei, mn, mm, target, t))
                    max_edges = max(max_edges, ei.shape[0])

                if not items:
                    continue

                for nf, ei, mn, mm, target, t in items:
                    if ei.shape[0] < max_edges:
                        pad = np.zeros((max_edges - ei.shape[0], 2), dtype=np.int64)
                        ei = np.vstack([ei, pad])
                    padded_target = np.zeros(args.max_moves, dtype=np.float32)
                    padded_target[:len(target)] = target

                    batch_nf.append(nf)
                    batch_ei.append(ei)
                    batch_mn.append(mn)
                    batch_mm.append(mm)
                    batch_targets.append(padded_target)
                    batch_t.append(t)

                nf_t = torch.tensor(np.stack(batch_nf), dtype=torch.float32, device=device)
                ei_t = torch.tensor(np.stack(batch_ei), dtype=torch.long, device=device)
                mn_t = torch.tensor(np.stack(batch_mn), dtype=torch.long, device=device)
                mm_t = torch.tensor(np.stack(batch_mm), dtype=torch.bool, device=device)
                targets_t = torch.tensor(np.stack(batch_targets), dtype=torch.float32, device=device)
                t_t = torch.tensor(batch_t, dtype=torch.float32, device=device)

                scores = model(nf_t, ei_t, mn_t, mm_t, t_t)

                log_probs = F.log_softmax(scores, dim=-1)
                log_probs = log_probs.masked_fill(~mm_t, 0.0)
                loss = -(targets_t * log_probs).sum(dim=-1).mean()

                pred_top = scores.argmax(dim=-1)
                target_top = targets_t.argmax(dim=-1)
                acc = (pred_top == target_top).float().mean().item()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                epoch_acc += acc
                pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{acc:.2%}")

            scheduler.step()
            print(f"  Epoch {epoch}: loss={epoch_loss/args.steps_per_epoch:.4f}, "
                  f"acc={epoch_acc/args.steps_per_epoch:.2%}")

        # Evaluate with greedy denoising
        print("  Evaluating (greedy)...")
        val_costs = []
        for vi in range(min(args.n_eval, len(val_instances))):
            _, cost = greedy_denoise(
                model, config, manifold, val_instances[vi],
                args.max_moves, n_denoise, device
            )
            val_costs.append(cost)
        avg_cost = np.mean(val_costs)
        ref = np.mean(val_targets[:len(val_costs)])
        gap = (avg_cost / abs(ref) - 1) * 100 if abs(ref) > 1e-8 else 0
        print(f"  Val (greedy): cost={avg_cost:.4f}, gap={gap:+.2f}% (ref={ref:.4f})")

        # Evaluate with best-of-K stochastic
        print(f"  Evaluating (best-of-{args.K})...")
        val_costs_bok = []
        for vi in range(min(args.n_eval, len(val_instances))):
            _, cost = best_of_k_denoise(
                model, config, manifold, val_instances[vi],
                args.max_moves, n_denoise, device, K=args.K,
            )
            val_costs_bok.append(cost)
        avg_bok = np.mean(val_costs_bok)
        gap_bok = (avg_bok / abs(ref) - 1) * 100 if abs(ref) > 1e-8 else 0
        print(f"  Val (best-of-{args.K}): cost={avg_bok:.4f}, gap={gap_bok:+.2f}%")

        # Checkpoint
        path = os.path.join(args.ckpt_dir, f'round{round_id}_{args.problem}{args.N}.pt')
        torch.save({
            'round': round_id, 'model_state_dict': model.state_dict(),
            'gap_greedy': gap, 'gap_bok': gap_bok,
            'cost_greedy': avg_cost, 'cost_bok': avg_bok,
            'problem': args.problem,
            'tau_min': tau_min, 'tau_max': tau_max,
            'args': vars(args),
        }, path)

        eval_gap = min(gap, gap_bok)
        if eval_gap < best_val_gap:
            best_val_gap = eval_gap
            best_path = os.path.join(args.ckpt_dir, f'best_{args.problem}{args.N}.pt')
            torch.save({
                'round': round_id, 'model_state_dict': model.state_dict(),
                'gap_greedy': gap, 'gap_bok': gap_bok,
                'problem': args.problem,
                'tau_min': tau_min, 'tau_max': tau_max,
                'args': vars(args),
            }, best_path)
            print(f"  [best] gap={eval_gap:+.2f}% -> {best_path}")

        # Self-play: improve targets via stochastic best-of-K
        if round_id < args.n_rounds:
            model.eval()
            train_targets, n_improved = selfplay_improve(
                model, config, manifold, instances, train_targets,
                args.max_moves, n_denoise, device,
                K=args.K, n_improve=args.n_improve,
            )
            if n_improved == 0:
                print("  No improvements. Stopping early.")
                break

    print(f"\nDone. Best val gap: {best_val_gap:+.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FMD self-play for any CO problem')
    parser.add_argument('--problem', type=str, required=True, choices=list(PROBLEM_CONFIGS.keys()))
    parser.add_argument('--N', type=int, default=20)
    parser.add_argument('--n_instances', type=int, default=2000)
    parser.add_argument('--n_val', type=int, default=50)
    parser.add_argument('--max_moves', type=int, default=1000)
    parser.add_argument('--n_rounds', type=int, default=5)
    parser.add_argument('--epochs_per_round', type=int, default=15)
    parser.add_argument('--steps_per_epoch', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--K', type=int, default=8, help='Best-of-K for stochastic sampling')
    parser.add_argument('--n_quality_levels', type=int, default=5)
    parser.add_argument('--n_denoise', type=int, default=None)
    parser.add_argument('--n_eval', type=int, default=20)
    parser.add_argument('--n_improve', type=int, default=None)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    parser.add_argument('--sub_tsp', type=str, default='2opt',
                        choices=['2opt', 'nn', 'reviser'],
                        help='Sub-TSP solver for CVRP: 2opt (default), nn (fast), reviser (GPU)')
    parser.add_argument('--reviser_path', type=str,
                        default='./pretrained/Reviser-stage2/reviser_10/epoch-299.pt',
                        help='Path to GLOP reviser checkpoint')
    args = parser.parse_args()
    train(args)
