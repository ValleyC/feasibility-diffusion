"""
State-Potential Diffusion on the Feasibility Manifold.

Instead of predicting the inverse move (high-variance one-hot target) or
the cost-improving move (not true diffusion), this trainer learns a
time-dependent scalar potential f_θ(x, t) ≈ log p_t(x).

Reverse transition logits are DERIVED from potential differences:
    logit(x → y) = f_θ(y, t-1) - f_θ(x, t)
for feasible neighbor y.

The potential captures: "how likely is state x at diffusion time t?"
  - At t=0: high potential for good tours, low for bad
  - At t=T: uniform potential (all tours equally likely)

Training:
  The potential is learned via denoising score matching. Given pairs
  (x_t, x_0) from the forward process, the model learns that x_0's
  neighbors should have higher potential than random neighbors at time
  t-1. This gives a contrastive signal without needing to predict the
  exact inverse move.

Architecture:
  - Same GNN backbone as before
  - But output is ONE SCALAR per state (global pool → MLP → scalar)
  - Plus the 4-node scorer to predict potential DIFFERENCES efficiently

Usage:
    python -m training.potential_trainer --problem tsp --N 50 --device cuda:0
"""

import sys, os, time, argparse, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from models.problem_configs import PROBLEM_CONFIGS
from training.generic_trainer import GNNLayer, prepare_batch_item
from core.forward import scramble_with_moves
from problems.tsp.data import generate_dataset


# ─── Model ──────────────────────────────────────────────────────

class PotentialScorer(nn.Module):
    """GNN that predicts potential DIFFERENCES between neighboring states.

    For a 2-opt move (i,j), the potential difference f(y) - f(x) depends
    on the 4 boundary nodes. The model predicts this difference directly.

    This is the discrete analog of the continuous score function:
      continuous: ∇_x log p_t(x)
      discrete:   f_t(y) - f_t(x) for neighbor y

    The reverse transition logit for move m is:
      logit(m) = potential_diff(m) = f(neighbor_m, t-1) - f(x, t)
    """

    def __init__(self, n_node_features, hidden_dim=128, n_layers=6):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.time_embed = nn.Sequential(
            nn.Linear(16, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.embed = nn.Sequential(
            nn.Linear(n_node_features, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gnn = nn.ModuleList([GNNLayer(hidden_dim) for _ in range(n_layers)])

        # Potential difference scorer: 4 boundary node embeddings → scalar
        self.diff_scorer = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def time_encoding(self, t):
        freqs = torch.exp(torch.arange(0, 8, device=t.device).float()
                          * (-np.log(10000) / 8))
        args = t.unsqueeze(-1) * freqs
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(self, node_features, edge_index, move_nodes, move_mask, t):
        """Predict potential difference for each move.

        Returns:
            (B, M) potential differences — positive means neighbor is more
            likely than current state under the reverse kernel.
        """
        B, N, _ = node_features.shape
        D = self.hidden_dim
        t_emb = self.time_embed(self.time_encoding(t))
        h = self.embed(node_features) + t_emb.unsqueeze(1)
        for layer in self.gnn:
            h = layer(h, edge_index)

        M = move_nodes.shape[1]
        idx = move_nodes.clamp(0, N - 1)
        h_parts = []
        for p in range(4):
            node_idx = idx[:, :, p].unsqueeze(-1).expand(-1, -1, D)
            h_parts.append(h.gather(1, node_idx))
        h_moves = torch.cat(h_parts, dim=-1)

        scores = self.diff_scorer(h_moves).squeeze(-1)
        return scores.masked_fill(~move_mask, float('-inf'))


# ─── Training data generation ──────────────────────────────────

def generate_potential_data(config, manifold, instances, clean_solutions,
                            t_max, max_moves, n_samples_per_instance=10):
    """Generate training data for potential learning.

    For each (x_0, instance), create forward trajectories and store
    (x_t, t, x_{t-1}) triplets. The model learns that x_{t-1} should
    have higher potential than x_t (since x_{t-1} is closer to x_0).

    Instead of one-hot inverse move, we store the FULL trajectory context:
    which moves bring x_t closer to x_0 (positive examples) vs which
    moves lead away (negative examples).
    """
    pool = []
    for idx in tqdm(range(len(instances)), desc="  Generating data"):
        inst = instances[idx]
        x_0 = clean_solutions[idx]

        for _ in range(n_samples_per_instance):
            t = np.random.randint(1, t_max + 1)

            # Forward: x_0 → x_1 → ... → x_t
            x_t, applied_moves = scramble_with_moves(manifold, x_0, inst, t)

            if not applied_moves:
                continue

            # The predecessor x_{t-1} is obtained by applying the inverse
            # of the last move. For 2-opt, the inverse IS the same move.
            last_move = applied_moves[-1]

            # x_{t-1} = apply(x_t, last_move) — 2-opt is self-inverse
            x_prev = manifold.apply_move(x_t, last_move)

            # Cost of x_t and x_{t-1}
            cost_t = manifold.cost(x_t, inst)
            cost_prev = manifold.cost(x_prev, inst)

            # Enumerate moves at x_t
            moves = manifold.enumerate_moves(x_t, inst)
            if not moves:
                continue

            # For each move, compute how it changes the cost
            # AND whether it leads toward x_{t-1} (the predecessor)
            # The move that leads to x_{t-1} should get highest potential
            deltas = np.array([manifold.move_delta(x_t, m, inst) for m in moves],
                              dtype=np.float32)

            # Find the inverse move index
            inv_idx = -1
            for i, m in enumerate(moves):
                if m == last_move:
                    inv_idx = i
                    break

            # Subsample if needed, always keeping inverse move
            if len(moves) > max_moves and inv_idx >= 0:
                other_indices = [i for i in range(len(moves)) if i != inv_idx]
                keep = sorted(np.random.choice(
                    other_indices, max_moves - 1, replace=False).tolist())
                # Insert inverse move at correct position
                insert_pos = 0
                for k, ki in enumerate(keep):
                    if ki > inv_idx:
                        insert_pos = k
                        break
                    insert_pos = k + 1
                keep.insert(insert_pos, inv_idx)
                moves = [moves[i] for i in keep]
                deltas = deltas[keep]
                inv_idx = insert_pos

            pool.append({
                'idx': idx,
                'x_t': x_t.copy() if hasattr(x_t, 'copy') else x_t,
                't': t,
                'moves': moves,
                'deltas': deltas,
                'inv_idx': inv_idx,
                'cost_t': cost_t,
                'cost_prev': cost_prev,
            })

    print(f"  {len(pool)} samples ({len(pool)/len(instances):.1f} per instance)")
    return pool


# ─── Inference ──────────────────────────────────────────────────

@torch.no_grad()
def potential_denoise(model, config, manifold, instance, max_moves, n_steps,
                      device, temperature=1.0):
    """Denoise by sampling from potential-derived reverse distribution."""
    model.eval()
    sol = manifold.sample_random(instance)
    best_sol = sol.copy() if hasattr(sol, 'copy') else sol
    best_cost = manifold.cost(sol, instance)

    for step in range(n_steps):
        moves = manifold.enumerate_moves(sol, instance)
        if not moves:
            break
        if len(moves) > max_moves:
            subset = np.random.choice(len(moves), max_moves, replace=False)
            moves = [moves[i] for i in sorted(subset)]

        t_val = 1.0 - step / max(n_steps - 1, 1)

        nf, ei, mn, mm, _ = prepare_batch_item(
            config, sol, instance, moves, 0, t_val, max_moves)

        nf_t = torch.tensor(nf, dtype=torch.float32, device=device).unsqueeze(0)
        ei_t = torch.tensor(ei, dtype=torch.long, device=device).unsqueeze(0)
        mn_t = torch.tensor(mn, dtype=torch.long, device=device).unsqueeze(0)
        mm_t = torch.tensor(mm, dtype=torch.bool, device=device).unsqueeze(0)
        t_t = torch.tensor([t_val], dtype=torch.float32, device=device)

        # Model predicts potential differences → reverse logits
        pot_diffs = model(nf_t, ei_t, mn_t, mm_t, t_t)

        # Sample from reverse distribution
        temp = max(temperature * t_val, 0.1)
        probs = F.softmax(pot_diffs.squeeze(0)[:len(moves)] / temp, dim=-1)
        pick = torch.multinomial(probs, 1).item()

        sol = manifold.apply_move(sol, moves[pick])
        cost = manifold.cost(sol, instance)
        if cost < best_cost:
            best_cost = cost
            best_sol = sol.copy() if hasattr(sol, 'copy') else sol

    return best_sol, best_cost


@torch.no_grad()
def best_of_k_potential(model, config, manifold, instance, max_moves,
                        n_steps, device, K=8, temperature=1.0):
    """Run K reverse trajectories, return best."""
    best_sol, best_cost = None, float('inf')
    for _ in range(K):
        sol, cost = potential_denoise(model, config, manifold, instance,
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
    t_max = max(N * 2, 50)

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

    # Validation
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
        pool = generate_potential_data(
            config, manifold, instances, clean_solutions,
            t_max, args.max_moves,
            n_samples_per_instance=args.n_samples_per_instance)
        with open(pool_path, 'wb') as f:
            pickle.dump(pool, f)

    # Model
    model = PotentialScorer(
        n_node_features=config.n_node_features,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(device)
    print(f"  PotentialScorer: {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    n_pool = len(pool)
    best_gap = float('inf')
    n_denoise = t_max

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        epoch_loss, epoch_acc = 0.0, 0.0

        pbar = tqdm(range(args.steps_per_epoch), desc=f"  Epoch {epoch}")
        for step in pbar:
            batch_nf, batch_ei, batch_mn, batch_mm = [], [], [], []
            batch_targets, batch_t = [], []
            max_edges = 0

            items = []
            for _ in range(args.batch_size):
                s = pool[np.random.randint(n_pool)]
                idx, x_t, t = s['idx'], s['x_t'], s['t']
                moves, deltas = s['moves'], s['deltas']
                inv_idx = s['inv_idx']

                if not moves or inv_idx < 0:
                    continue

                t_norm = t / t_max

                # Build soft target: combine inverse move signal + cost signal
                # The inverse move should get high probability (trajectory info)
                # Moves with negative delta should also get some credit (cost info)
                # This blends the two signals with a mixing weight
                n_moves = len(moves)

                # Component 1: inverse move (trajectory reversal)
                inv_logits = np.full(n_moves, -5.0, dtype=np.float32)
                inv_logits[inv_idx] = 5.0

                # Component 2: cost-based (Boltzmann on deltas)
                tau = max(np.median(np.abs(deltas)) * 0.5, 0.01)
                cost_logits = -deltas / tau

                # Blend: more trajectory signal at high t (noisy), more cost at low t
                alpha = min(t_norm * 2, 1.0)  # 0→1 as t increases
                blended_logits = alpha * inv_logits + (1 - alpha) * cost_logits

                # Softmax target
                blended_logits -= blended_logits.max()
                target = np.exp(blended_logits)
                target = (target / target.sum()).astype(np.float32)

                nf, ei, mn, mm, _ = prepare_batch_item(
                    config, x_t, instances[idx], moves, inv_idx,
                    t_norm, args.max_moves)
                items.append((nf, ei, mn, mm, target, t_norm))
                max_edges = max(max_edges, ei.shape[0])

            if not items:
                continue

            for nf, ei, mn, mm, target, t in items:
                if ei.shape[0] < max_edges:
                    pad = np.zeros((max_edges - ei.shape[0], 2), dtype=np.int64)
                    ei = np.vstack([ei, pad])
                padded = np.zeros(args.max_moves, dtype=np.float32)
                padded[:len(target)] = target
                batch_nf.append(nf)
                batch_ei.append(ei)
                batch_mn.append(mn)
                batch_mm.append(mm)
                batch_targets.append(padded)
                batch_t.append(t)

            nf_t = torch.tensor(np.stack(batch_nf), dtype=torch.float32, device=device)
            ei_t = torch.tensor(np.stack(batch_ei), dtype=torch.long, device=device)
            mn_t = torch.tensor(np.stack(batch_mn), dtype=torch.long, device=device)
            mm_t = torch.tensor(np.stack(batch_mm), dtype=torch.bool, device=device)
            targets_t = torch.tensor(np.stack(batch_targets), dtype=torch.float32, device=device)
            t_t = torch.tensor(batch_t, dtype=torch.float32, device=device)

            scores = model(nf_t, ei_t, mn_t, mm_t, t_t)

            # Cross-entropy with soft blended target
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

        # Evaluate
        if epoch % args.eval_freq == 0 or epoch == args.n_epochs:
            model.eval()
            eval_single, eval_bok = [], []
            n_eval = min(args.n_eval, len(val_instances))

            eval_pbar = tqdm(range(n_eval), desc="  Evaluating")
            for vi in eval_pbar:
                _, c_s = potential_denoise(model, config, manifold,
                    val_instances[vi], args.max_moves, n_denoise, device)
                eval_single.append(c_s)
                _, c_b = best_of_k_potential(model, config, manifold,
                    val_instances[vi], args.max_moves, n_denoise, device, K=args.K)
                eval_bok.append(c_b)
                eval_pbar.set_postfix(single=f"{np.mean(eval_single):.3f}",
                                      bok=f"{np.mean(eval_bok):.3f}")

            avg_s = np.mean(eval_single)
            avg_b = np.mean(eval_bok)
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
    parser = argparse.ArgumentParser(description='State-Potential Diffusion Trainer')
    parser.add_argument('--problem', type=str, default='tsp',
                        choices=list(PROBLEM_CONFIGS.keys()))
    parser.add_argument('--N', type=int, default=50)
    parser.add_argument('--n_instances', type=int, default=5000)
    parser.add_argument('--n_val', type=int, default=100)
    parser.add_argument('--max_moves', type=int, default=1200)
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
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/potential')
    args = parser.parse_args()
    train(args)
