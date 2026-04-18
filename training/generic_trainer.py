"""
Generic trainer for feasibility-preserving diffusion on ANY CO problem.

Works with any FeasibilityManifold + ProblemConfig. The training loop is
problem-agnostic; only the feature extraction (via ProblemConfig) is
problem-specific.

Training: greedy local-search trajectories generate (solution, best_move) pairs.
Model: GNN on problem graph + 4-node MLP move scorer (shared across problems).
Inference: greedy denoising from random feasible solution.

Usage:
    python -m training.generic_trainer --problem tsp --N 50 --device cuda:0
    python -m training.generic_trainer --problem cvrp --N 20 --device cuda:0
    python -m training.generic_trainer --problem mis --N 50 --device cuda:0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import multiprocessing as mp
from multiprocessing import cpu_count

from models.problem_configs import PROBLEM_CONFIGS


# ─── Model (shared across all problems) ─────────────────────

class GNNLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.msg = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim))
        self.upd = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
                                 nn.Linear(hidden_dim, hidden_dim))
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index):
        B, N, D = h.shape
        if edge_index.shape[1] == 0:
            return h
        src = edge_index[:, :, 0].clamp(0, N - 1)
        tgt = edge_index[:, :, 1].clamp(0, N - 1)
        h_src = h.gather(1, src.unsqueeze(-1).expand(-1, -1, D))
        h_tgt = h.gather(1, tgt.unsqueeze(-1).expand(-1, -1, D))
        msg = self.msg(torch.cat([h_tgt, h_src], dim=-1))
        agg = torch.zeros_like(h)
        agg.scatter_add_(1, tgt.unsqueeze(-1).expand(-1, -1, D), msg)
        return self.norm(h + self.upd(torch.cat([h, agg], dim=-1)))


class GenericMoveScorer(nn.Module):
    def __init__(self, n_node_features, hidden_dim=64, n_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Sequential(nn.Linear(n_node_features, hidden_dim), nn.SiLU(),
                                   nn.Linear(hidden_dim, hidden_dim))
        self.gnn = nn.ModuleList([GNNLayer(hidden_dim) for _ in range(n_layers)])
        self.scorer = nn.Sequential(nn.Linear(4 * hidden_dim, hidden_dim), nn.SiLU(),
                                    nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                    nn.Linear(hidden_dim, 1))

    def forward(self, node_features, edge_index, move_nodes, move_mask):
        B, N, _ = node_features.shape
        D = self.hidden_dim
        h = self.embed(node_features)
        for layer in self.gnn:
            h = layer(h, edge_index)
        M = move_nodes.shape[1]
        idx = move_nodes.clamp(0, N - 1).unsqueeze(-1).expand(-1, -1, -1, D)
        h_exp = h.unsqueeze(1).expand(-1, M, -1, -1)
        h_moves = h_exp.gather(2, idx).reshape(B, M, 4 * D)
        scores = self.scorer(h_moves).squeeze(-1)
        return scores.masked_fill(~move_mask, float('-inf'))


# ─── Data generation ─────────────────────────────────────────

def _process_one_instance(args):
    """Process a single instance (for multiprocessing).

    Returns LIGHTWEIGHT samples: (idx, solution_copy, best_move, iteration).
    Does NOT return the full moves list (too large for pipe serialization).
    Moves are re-enumerated during batch preparation.
    """
    idx, inst, manifold_class, max_moves, max_iters, n_restarts = args

    if isinstance(manifold_class, type):
        manifold = manifold_class()
    else:
        manifold = manifold_class

    samples = []  # lightweight: (idx, solution, best_move, iteration)
    best_cost = float('inf')
    time_limit = 120  # seconds per instance — prevents hangs on outliers
    t_start = time.time()

    for restart in range(n_restarts):
        if time.time() - t_start > time_limit:
            break

        sol = manifold.sample_random(inst)

        moves_init = manifold.enumerate_moves(sol, inst)
        n_degrade = np.random.randint(0, max(len(moves_init) // 3, 1) + 1)
        for _ in range(n_degrade):
            mv = manifold.enumerate_moves(sol, inst)
            if len(mv) == 0:
                break
            sol = manifold.apply_move(sol, mv[np.random.randint(len(mv))])

        for iteration in range(max_iters):
            if time.time() - t_start > time_limit:
                break
            moves = manifold.enumerate_moves(sol, inst)
            if len(moves) == 0 or len(moves) > max_moves:
                break
            deltas = np.array([manifold.move_delta(sol, m, inst) for m in moves])
            best_move_idx = int(np.argmin(deltas))
            if deltas[best_move_idx] >= -1e-10:
                break
            best_move = moves[best_move_idx]
            samples.append((idx, sol.copy() if hasattr(sol, 'copy') else sol,
                           best_move, iteration))
            sol = manifold.apply_move(sol, best_move)

        c = manifold.cost(sol, inst)
        best_cost = min(best_cost, c)

    return samples, best_cost


def build_sample_pool(config, manifold, instances, max_moves, max_iters=100,
                      n_restarts=3, n_workers=None):
    """Build training samples from greedy improvement trajectories.

    Uses multiprocessing for parallel generation across instances.
    """
    if n_workers is None:
        n_workers = min(cpu_count(), 64)

    # For multiprocessing: pass manifold CLASS (not instance) to avoid pickle issues
    # Each worker creates a fresh manifold with built-in 2-opt (no GPU sub_solver)
    manifold_class = type(manifold)

    work_args = [
        (idx, inst, manifold_class, max_moves, max_iters, n_restarts)
        for idx, inst in enumerate(instances)
    ]

    sample_pool = []
    costs = []

    if n_workers <= 1:
        # Serial — use the original manifold (may have sub_solver)
        work_args_serial = [
            (idx, inst, manifold, max_moves, max_iters, n_restarts)
            for idx, inst in enumerate(instances)
        ]
        pbar = tqdm(work_args_serial, desc="  Pool generation")
        for args in pbar:
            samples, cost = _process_one_instance(args)
            sample_pool.extend(samples)
            costs.append(cost)
            pbar.set_postfix(samples=len(sample_pool), cost=f"{cost:.2f}")
    else:
        # Parallel — process results as they arrive (not collected into list)
        # to avoid pipe backpressure that causes BrokenPipeError
        print(f"  Pool generation: {len(instances)} instances × {n_restarts} restarts, "
              f"{n_workers} workers")
        with mp.Pool(n_workers) as p:
            for samples, cost in tqdm(
                p.imap(_process_one_instance, work_args, chunksize=1),
                total=len(instances),
                desc="  Pool generation",
            ):
                sample_pool.extend(samples)
                costs.append(cost)

    return sample_pool, costs


def prepare_batch_item(config, solution, instance, moves, best_idx, progress, max_moves):
    """Prepare tensors for one training sample."""
    node_features = config.build_node_features(solution, instance, progress)
    edge_index = config.build_edges(solution, instance)

    move_nodes = np.zeros((max_moves, 4), dtype=np.int64)
    move_mask = np.zeros(max_moves, dtype=bool)

    for i, m in enumerate(moves[:max_moves]):
        a, b, c, d = config.move_to_4nodes(solution, m, instance)
        move_nodes[i] = [a, b, c, d]
        move_mask[i] = True

    label = min(best_idx, max_moves - 1)
    return node_features, edge_index, move_nodes, move_mask, label


# ─── Inference ───────────────────────────────────────────────

@torch.no_grad()
def greedy_denoise(model, config, manifold, instance, max_moves, n_steps, device):
    model.eval()
    sol = manifold.sample_random(instance)

    for step in range(n_steps):
        moves = manifold.enumerate_moves(sol, instance)
        if len(moves) == 0 or len(moves) > max_moves:
            break

        n_total = len([m for m in range(len(moves))])
        progress = step / max(n_steps - 1, 1)
        nf, ei, mn, mm, _ = prepare_batch_item(
            config, sol, instance, moves, 0, progress, max_moves
        )

        nf_t = torch.tensor(nf, dtype=torch.float32, device=device).unsqueeze(0)
        ei_t = torch.tensor(ei, dtype=torch.long, device=device).unsqueeze(0)
        mn_t = torch.tensor(mn, dtype=torch.long, device=device).unsqueeze(0)
        mm_t = torch.tensor(mm, dtype=torch.bool, device=device).unsqueeze(0)

        scores = model(nf_t, ei_t, mn_t, mm_t)
        best = scores.argmax(dim=-1).item()

        if best < len(moves):
            d = manifold.move_delta(sol, moves[best], instance)
            if d < 0:
                sol = manifold.apply_move(sol, moves[best])

    cost = manifold.cost(sol, instance)
    return sol, cost


# ─── Main ────────────────────────────────────────────────────

def train(args):
    device = torch.device(args.device)

    # Problem setup
    ConfigClass = PROBLEM_CONFIGS[args.problem]
    config = ConfigClass()
    manifold = config.create_manifold()
    print(f"Problem: {config.name}, N={args.N}")

    # Generate instances
    print(f"Generating {args.n_instances} training instances...")
    instances = [config.create_instance(args.N, seed=42 + i) for i in range(args.n_instances)]
    val_instances = [config.create_instance(args.N, seed=99999 + i) for i in range(args.n_val)]

    # Build training pool
    print("Building training sample pool...")
    t0 = time.time()
    pool, train_costs = build_sample_pool(config, manifold, instances, args.max_moves)
    print(f"  {len(pool)} samples in {time.time() - t0:.1f}s, "
          f"avg converged cost: {np.mean(train_costs):.4f}")

    _, val_costs = build_sample_pool(config, manifold, val_instances, args.max_moves)

    # Model
    model = GenericMoveScorer(
        n_node_features=config.n_node_features,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GenericMoveScorer: {n_params:,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_gap = float('inf')
    n_pool = len(pool)
    n_denoise = args.n_denoise if args.n_denoise else max(args.N * 3, 50)

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        epoch_loss, epoch_acc = 0.0, 0.0

        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch}")
        for step in pbar:
            batch_nf, batch_ei, batch_mn, batch_mm, batch_labels = [], [], [], [], []
            max_edges = 0

            items = []
            for _ in range(args.batch_size):
                s_idx = np.random.randint(n_pool)
                inst_idx, sol, best_move, step_i = pool[s_idx]
                total_steps = max(
                    len([p for p in pool if p[0] == inst_idx]) - 1, 1
                )
                progress = step_i / total_steps

                # Re-enumerate moves from solution (lightweight pool doesn't store them)
                moves = manifold.enumerate_moves(sol, instances[inst_idx])
                # Find index of the stored best_move
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
        print(f"Epoch {epoch}: loss={epoch_loss / args.steps_per_epoch:.4f}, "
              f"acc={epoch_acc / args.steps_per_epoch:.2%}")

        if epoch % args.eval_freq == 0:
            eval_costs = []
            for idx in range(min(args.n_eval, len(val_instances))):
                _, cost = greedy_denoise(
                    model, config, manifold, val_instances[idx],
                    args.max_moves, n_denoise, device
                )
                eval_costs.append(cost)

            avg_cost = np.mean(eval_costs)
            ref = np.mean(val_costs[:len(eval_costs)])
            gap = (avg_cost / abs(ref) - 1) * 100 if abs(ref) > 1e-8 else 0
            print(f"  Eval: cost={avg_cost:.4f}, gap={gap:.2f}% (ref={ref:.4f})")

            if gap < best_gap:
                best_gap = gap
                path = os.path.join(args.ckpt_dir, f'best_{args.problem}{args.N}.pt')
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'gap': gap, 'cost': avg_cost, 'problem': args.problem,
                    'args': vars(args),
                }, path)
                print(f"  [checkpoint] best gap={gap:.2f}%")

    print(f"\nDone. Best gap: {best_gap:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--problem', type=str, required=True, choices=list(PROBLEM_CONFIGS.keys()))
    parser.add_argument('--N', type=int, default=20)
    parser.add_argument('--n_instances', type=int, default=2000)
    parser.add_argument('--n_val', type=int, default=50)
    parser.add_argument('--max_moves', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--steps_per_epoch', type=int, default=200)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--n_denoise', type=int, default=None)
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--n_eval', type=int, default=20)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    args = parser.parse_args()
    train(args)
