"""
Generic self-play trainer for ANY CO problem.

Same AlphaZero principle as TSP self-play, but works with any FeasibilityManifold:
  Round 1: Train on greedy improvement trajectories
  Round 2+: Best-of-K inference → upgrade targets → retrain

Usage:
    python -m training.generic_selfplay --problem cvrp --N 50 --device cuda:0
    python -m training.generic_selfplay --problem tsp --N 50 --device cuda:0
    python -m training.generic_selfplay --problem pctsp --N 20 --device cuda:0
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
    prepare_batch_item, greedy_denoise,
)


def selfplay_improve(model, config, manifold, instances, targets,
                     max_moves, n_steps, device, K=8, n_improve=None):
    """Best-of-K inference on instances, upgrade targets where improved."""
    if n_improve is None:
        n_improve = len(instances)
    n_improve = min(n_improve, len(instances))

    new_targets = list(targets)
    n_improved = 0

    pbar = tqdm(range(n_improve), desc="  Self-play", leave=False)
    for idx in pbar:
        old_cost = targets[idx]

        best_cost = float('inf')
        for _ in range(K):
            sol = manifold.sample_random(instances[idx])

            # Denoising with stochastic exploration
            for step in range(n_steps):
                moves = manifold.enumerate_moves(sol, instances[idx])
                if len(moves) == 0 or len(moves) > max_moves:
                    break

                progress = step / max(n_steps - 1, 1)
                nf, ei, mn, mm, _ = prepare_batch_item(
                    config, sol, instances[idx], moves, 0, progress, max_moves
                )

                nf_t = torch.tensor(nf, dtype=torch.float32, device=device).unsqueeze(0)
                ei_t = torch.tensor(ei, dtype=torch.long, device=device).unsqueeze(0)
                mn_t = torch.tensor(mn, dtype=torch.long, device=device).unsqueeze(0)
                mm_t = torch.tensor(mm, dtype=torch.bool, device=device).unsqueeze(0)

                with torch.no_grad():
                    scores = model(nf_t, ei_t, mn_t, mm_t)

                if K > 1 and progress < 0.7:
                    probs = F.softmax(scores.squeeze(0) / 0.5, dim=-1)
                    best_idx = torch.multinomial(probs, 1).item()
                else:
                    best_idx = scores.argmax(dim=-1).item()

                if best_idx < len(moves):
                    d = manifold.move_delta(sol, moves[best_idx], instances[idx])
                    if d < 0:
                        sol = manifold.apply_move(sol, moves[best_idx])

            cost = manifold.cost(sol, instances[idx])
            best_cost = min(best_cost, cost)

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

    # Optional: use batched sub-TSP solver for CVRP-family problems
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

    # Build initial targets BEFORE creating model on GPU
    # (multiprocessing.Pool + CUDA fork = deadlock)
    print("Building initial targets...")
    _, train_targets = build_sample_pool(
        config, manifold, instances, args.max_moves, n_restarts=3
    )
    _, val_targets = build_sample_pool(
        config, manifold, val_instances, args.max_moves, n_restarts=3
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

        # Build training samples (with degradation for diversity)
        print("Building training samples...")
        t0 = time.time()
        # After round 1, model is on GPU — use serial to avoid CUDA fork deadlock
        pool, _ = build_sample_pool(
            config, manifold, instances, args.max_moves, n_restarts=3,
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
                batch_nf, batch_ei, batch_mn, batch_mm, batch_labels = [], [], [], [], []
                max_edges = 0

                items = []
                for _ in range(args.batch_size):
                    s_idx = np.random.randint(n_pool)
                    inst_idx, sol, best_move, step_i = pool[s_idx]
                    total = max(len([p for p in pool if p[0] == inst_idx]) - 1, 1)
                    progress = step_i / total

                    moves = manifold.enumerate_moves(sol, instances[inst_idx])
                    best_idx = 0
                    for mi, m in enumerate(moves):
                        if m == best_move:
                            best_idx = mi
                            break

                    nf, ei, mn, mm, label = prepare_batch_item(
                        config, sol, instances[inst_idx], moves, best_idx,
                        progress, args.max_moves
                    )
                    items.append((nf, ei, mn, mm, label))
                    max_edges = max(max_edges, ei.shape[0])

                for nf, ei, mn, mm, label in items:
                    if ei.shape[0] < max_edges:
                        pad = np.zeros((max_edges - ei.shape[0], 2), dtype=np.int64)
                        ei = np.vstack([ei, pad])
                    batch_nf.append(nf)
                    batch_ei.append(ei)
                    batch_mn.append(mn)
                    batch_mm.append(mm)
                    batch_labels.append(label)

                nf_t = torch.tensor(np.stack(batch_nf), dtype=torch.float32, device=device)
                ei_t = torch.tensor(np.stack(batch_ei), dtype=torch.long, device=device)
                mn_t = torch.tensor(np.stack(batch_mn), dtype=torch.long, device=device)
                mm_t = torch.tensor(np.stack(batch_mm), dtype=torch.bool, device=device)
                labels_t = torch.tensor(batch_labels, dtype=torch.long, device=device)

                scores = model(nf_t, ei_t, mn_t, mm_t)
                loss = F.cross_entropy(scores, labels_t)
                acc = (scores.argmax(-1) == labels_t).float().mean().item()

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

        # Evaluate
        print("  Evaluating...")
        val_costs = []
        for idx in range(min(args.n_eval, len(val_instances))):
            _, cost = greedy_denoise(
                model, config, manifold, val_instances[idx],
                args.max_moves, n_denoise, device
            )
            val_costs.append(cost)
        avg_cost = np.mean(val_costs)
        ref = np.mean(val_targets[:len(val_costs)])
        gap = (avg_cost / abs(ref) - 1) * 100 if abs(ref) > 1e-8 else 0
        print(f"  Val: cost={avg_cost:.4f}, gap={gap:+.2f}% (ref={ref:.4f})")

        # Checkpoint
        path = os.path.join(args.ckpt_dir, f'round{round_id}_{args.problem}{args.N}.pt')
        torch.save({
            'round': round_id, 'model_state_dict': model.state_dict(),
            'gap': gap, 'cost': avg_cost, 'problem': args.problem,
            'args': vars(args),
        }, path)

        if gap < best_val_gap:
            best_val_gap = gap
            best_path = os.path.join(args.ckpt_dir, f'best_{args.problem}{args.N}.pt')
            torch.save({
                'round': round_id, 'model_state_dict': model.state_dict(),
                'gap': gap, 'cost': avg_cost, 'problem': args.problem,
                'args': vars(args),
            }, best_path)
            print(f"  [best] gap={gap:+.2f}% -> {best_path}")

        # Self-play: improve targets
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
    parser = argparse.ArgumentParser(description='Generic self-play for any CO problem')
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
    parser.add_argument('--K', type=int, default=8, help='Best-of-K for self-play')
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
