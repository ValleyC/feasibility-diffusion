"""
True Reverse Diffusion Trainer on the Feasibility Manifold.

The model learns the REVERSE of the forward random walk — predicting
which move was applied to corrupt x_{t-1} into x_t. This is genuine
discrete denoising diffusion, not annealed local search.

Key difference from the Boltzmann trainer:
  - Boltzmann: target = softmax(-cost_delta/tau) → "which move improves cost?"
  - Reverse:   target = inverse move from forward trajectory → "which move undoes corruption?"

The cost enters ONLY through x_0 (train on good solutions from solver/self-play).
The training target is the actual reverse transition, not cost improvement.

Usage:
    python -m training.reverse_diffusion_trainer --problem tsp --N 50 --device cuda:0
"""

import sys, os, time, argparse, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.problem_configs import PROBLEM_CONFIGS
from training.generic_trainer import GenericMoveScorer, GNNLayer, prepare_batch_item
from core.forward import sample_reverse_training_data
from problems.tsp.data import generate_dataset


# ─── Data generation ────────────────────────────────────────────

def generate_training_data(config, manifold, instances, clean_solutions,
                           t_max, max_moves, n_samples_per_instance=5):
    """Generate (x_t, t, inverse_move_idx) training data from forward trajectories.

    For each instance:
      1. Take clean solution x_0
      2. For each noise level: corrupt x_0 → x_t, record inverse move
      3. Store (instance_idx, x_t, t, inverse_move_idx)
    """
    pool = []
    for idx in tqdm(range(len(instances)), desc="  Generating trajectories"):
        inst = instances[idx]
        x_0 = clean_solutions[idx]

        for _ in range(n_samples_per_instance):
            t = np.random.randint(1, t_max + 1)
            x_t, t_actual, inv_idx, moves = sample_reverse_training_data(
                manifold, x_0, inst, t_max, t=t)

            if inv_idx < 0 or not moves:
                continue

            # Truncate moves to max_moves
            n_moves = len(moves)
            if n_moves > max_moves:
                # If inverse move is beyond max_moves, skip
                if inv_idx >= max_moves:
                    continue
                moves = moves[:max_moves]
                n_moves = max_moves

            pool.append((idx, x_t, t_actual, inv_idx, moves))

    print(f"  {len(pool)} training samples generated")
    return pool


# ─── Inference ──────────────────────────────────────────────────

@torch.no_grad()
def reverse_denoise(model, config, manifold, instance, max_moves, n_steps,
                    device, temperature=1.0):
    """Denoise by sampling from the learned reverse distribution.

    Unlike greedy local search, this allows worsening moves when the
    reverse kernel says so — essential for true reverse diffusion.
    """
    model.eval()
    sol = manifold.sample_random(instance)
    best_sol = sol.copy() if hasattr(sol, 'copy') else sol
    best_cost = manifold.cost(sol, instance)

    for step in range(n_steps):
        moves = manifold.enumerate_moves(sol, instance)
        if not moves:
            break
        n_use = min(len(moves), max_moves)
        moves = moves[:n_use]

        t_val = 1.0 - step / max(n_steps - 1, 1)

        nf, ei, mn, mm, _ = prepare_batch_item(
            config, sol, instance, moves, 0, t_val, max_moves)

        nf_t = torch.tensor(nf, dtype=torch.float32, device=device).unsqueeze(0)
        ei_t = torch.tensor(ei, dtype=torch.long, device=device).unsqueeze(0)
        mn_t = torch.tensor(mn, dtype=torch.long, device=device).unsqueeze(0)
        mm_t = torch.tensor(mm, dtype=torch.bool, device=device).unsqueeze(0)
        t_t = torch.tensor([t_val], dtype=torch.float32, device=device)

        scores = model(nf_t, ei_t, mn_t, mm_t, t_t)

        # Sample from the learned reverse distribution (not argmax!)
        # Temperature controls exploration: high t → more random, low t → more focused
        temp = max(temperature * t_val, 0.05)
        probs = F.softmax(scores.squeeze(0)[:n_use] / temp, dim=-1)
        pick = torch.multinomial(probs, 1).item()

        # Apply unconditionally — true reverse diffusion allows worsening moves
        sol = manifold.apply_move(sol, moves[pick])

        cost = manifold.cost(sol, instance)
        if cost < best_cost:
            best_cost = cost
            best_sol = sol.copy() if hasattr(sol, 'copy') else sol

    return best_sol, best_cost


@torch.no_grad()
def best_of_k_reverse(model, config, manifold, instance, max_moves, n_steps,
                      device, K=8, temperature=1.0):
    """Run K reverse diffusion trajectories, return best."""
    best_sol, best_cost = None, float('inf')
    for _ in range(K):
        sol, cost = reverse_denoise(model, config, manifold, instance,
                                    max_moves, n_steps, device, temperature)
        if cost < best_cost:
            best_cost = cost
            best_sol = sol
    return best_sol, best_cost


# ─── Training ──────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    ConfigClass = PROBLEM_CONFIGS[args.problem]
    config = ConfigClass()
    manifold = config.create_manifold()
    N = args.N
    t_max = max(N * 2, 50)  # max corruption steps

    print(f"Problem: {config.name}, N={N}, T_max={t_max}")

    # Generate instances + clean solutions
    cache_path = os.path.join(args.ckpt_dir, f'dataset_{args.problem}{N}.pkl')
    if os.path.exists(cache_path):
        print(f"Loading dataset from {cache_path}")
        with open(cache_path, 'rb') as f:
            instances, clean_solutions, clean_costs = pickle.load(f)
    else:
        print(f"Generating {args.n_instances} instances with 2-opt solutions...")
        coords_list, dist_list, tour_list, cost_list = generate_dataset(
            N, args.n_instances, seed=42, solver_restarts=5)
        instances = [{'coords': c, 'dist': d, 'N': N}
                     for c, d in zip(coords_list, dist_list)]
        clean_solutions = tour_list
        clean_costs = cost_list
        with open(cache_path, 'wb') as f:
            pickle.dump((instances, clean_solutions, clean_costs), f)
    print(f"  {len(instances)} instances, avg clean cost: {np.mean(clean_costs):.4f}")

    # Validation set
    val_coords, val_dist, val_tours, val_costs = generate_dataset(
        N, args.n_val, seed=99999, solver_restarts=5)
    val_instances = [{'coords': c, 'dist': d, 'N': N}
                     for c, d in zip(val_coords, val_dist)]
    ref_cost = np.mean(val_costs)
    print(f"  Val ref (2-opt): {ref_cost:.4f}")

    # Generate training pool
    pool_path = os.path.join(args.ckpt_dir, f'pool_{args.problem}{N}.pkl')
    if os.path.exists(pool_path):
        print(f"Loading pool from {pool_path}")
        with open(pool_path, 'rb') as f:
            pool = pickle.load(f)
        print(f"  {len(pool)} samples")
    else:
        pool = generate_training_data(
            config, manifold, instances, clean_solutions,
            t_max, args.max_moves,
            n_samples_per_instance=args.n_samples_per_instance)
        with open(pool_path, 'wb') as f:
            pickle.dump(pool, f)

    # Model
    model = GenericMoveScorer(
        n_node_features=config.n_node_features,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(device)
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    n_pool = len(pool)
    best_gap = float('inf')
    n_denoise = max(N * 3, 100)

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        epoch_loss, epoch_acc = 0.0, 0.0

        pbar = tqdm(range(args.steps_per_epoch), desc=f"  Epoch {epoch}")
        for step in pbar:
            batch_nf, batch_ei, batch_mn, batch_mm = [], [], [], []
            batch_labels, batch_t = [], []
            max_edges = 0

            items = []
            for _ in range(args.batch_size):
                s_idx = np.random.randint(n_pool)
                idx, x_t, t, inv_idx, moves = pool[s_idx]

                if inv_idx < 0 or not moves:
                    continue

                t_norm = t / t_max  # normalize to [0, 1]

                nf, ei, mn, mm, _ = prepare_batch_item(
                    config, x_t, instances[idx], moves, inv_idx, t_norm, args.max_moves)
                items.append((nf, ei, mn, mm, inv_idx, t_norm))
                max_edges = max(max_edges, ei.shape[0])

            if not items:
                continue

            for nf, ei, mn, mm, label, t in items:
                if ei.shape[0] < max_edges:
                    pad = np.zeros((max_edges - ei.shape[0], 2), dtype=np.int64)
                    ei = np.vstack([ei, pad])
                batch_nf.append(nf)
                batch_ei.append(ei)
                batch_mn.append(mn)
                batch_mm.append(mm)
                batch_labels.append(label)
                batch_t.append(t)

            nf_t = torch.tensor(np.stack(batch_nf), dtype=torch.float32, device=device)
            ei_t = torch.tensor(np.stack(batch_ei), dtype=torch.long, device=device)
            mn_t = torch.tensor(np.stack(batch_mn), dtype=torch.long, device=device)
            mm_t = torch.tensor(np.stack(batch_mm), dtype=torch.bool, device=device)
            labels_t = torch.tensor(batch_labels, dtype=torch.long, device=device)
            t_t = torch.tensor(batch_t, dtype=torch.float32, device=device)

            scores = model(nf_t, ei_t, mn_t, mm_t, t_t)

            # Cross-entropy on inverse move index — NOT Boltzmann on deltas
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
        if epoch % args.eval_freq == 0 or epoch == args.n_epochs:
            model.eval()
            eval_costs_single, eval_costs_bok = [], []
            n_eval = min(args.n_eval, len(val_instances))

            eval_pbar = tqdm(range(n_eval), desc="  Evaluating")
            for vi in eval_pbar:
                _, c_s = reverse_denoise(model, config, manifold, val_instances[vi],
                                         args.max_moves, n_denoise, device)
                eval_costs_single.append(c_s)
                _, c_b = best_of_k_reverse(model, config, manifold, val_instances[vi],
                                           args.max_moves, n_denoise, device, K=args.K)
                eval_costs_bok.append(c_b)
                eval_pbar.set_postfix(single=f"{np.mean(eval_costs_single):.3f}",
                                      bok=f"{np.mean(eval_costs_bok):.3f}")

            avg_s = np.mean(eval_costs_single)
            avg_b = np.mean(eval_costs_bok)
            gap_s = (avg_s / ref_cost - 1) * 100
            gap_b = (avg_b / ref_cost - 1) * 100
            print(f"  Val single:    {avg_s:.4f} ({gap_s:+.2f}% vs 2-opt)")
            print(f"  Val best-of-{args.K}: {avg_b:.4f} ({gap_b:+.2f}% vs 2-opt)")

            eval_gap = min(gap_s, gap_b)
            if eval_gap < best_gap:
                best_gap = eval_gap
                path = os.path.join(args.ckpt_dir, f'best_{args.problem}{N}.pt')
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'gap_single': gap_s, 'gap_bok': gap_b,
                    'ref_cost': ref_cost, 'args': vars(args),
                }, path)
                print(f"  [best] gap={eval_gap:+.2f}% → {path}")

    print(f"\nDone. Best gap: {best_gap:+.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='True Reverse Diffusion Trainer')
    parser.add_argument('--problem', type=str, default='tsp',
                        choices=list(PROBLEM_CONFIGS.keys()))
    parser.add_argument('--N', type=int, default=50)
    parser.add_argument('--n_instances', type=int, default=5000)
    parser.add_argument('--n_val', type=int, default=100)
    parser.add_argument('--max_moves', type=int, default=500)
    parser.add_argument('--n_samples_per_instance', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--steps_per_epoch', type=int, default=500)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--K', type=int, default=8)
    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--n_eval', type=int, default=50)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/reverse_diffusion')
    args = parser.parse_args()
    train(args)
