"""
Feasibility Manifold Diffusion (FMD) trainer for ANY CO problem.

CTMC diffusion on the feasibility manifold with Boltzmann score matching.
The model learns to predict move quality (score) from soft Boltzmann targets
derived from actual move deltas. At inference, the model replaces expensive
delta computation with a single GNN forward pass.

Label-free: no optimal solutions needed. Self-play bootstraps quality.

Training: solutions at various quality levels → compute deltas → Boltzmann
  soft targets → train GNN to predict scores.
Inference: start from random feasible solution → GNN scores moves → sample
  or argmax → apply if improving → repeat.

Usage:
    python -m training.generic_trainer --problem tsp --N 50 --device cuda:0
    python -m training.generic_trainer --problem cvrp --N 20 --device cuda:0
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


# ─── Model ──────────────────────────────────────────────────────

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
    """GNN move scorer with sinusoidal time conditioning for FMD."""

    def __init__(self, n_node_features, hidden_dim=64, n_layers=4):
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
        self.scorer = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def time_encoding(self, t):
        freqs = torch.exp(torch.arange(0, 8, device=t.device).float() * (-np.log(10000) / 8))
        args = t.unsqueeze(-1) * freqs
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def forward(self, node_features, edge_index, move_nodes, move_mask, t):
        B, N, _ = node_features.shape
        D = self.hidden_dim
        t_emb = self.time_embed(self.time_encoding(t))
        h = self.embed(node_features) + t_emb.unsqueeze(1)
        for layer in self.gnn:
            h = layer(h, edge_index)
        M = move_nodes.shape[1]
        idx = move_nodes.clamp(0, N - 1).unsqueeze(-1).expand(-1, -1, -1, D)
        h_exp = h.unsqueeze(1).expand(-1, M, -1, -1)
        h_moves = h_exp.gather(2, idx).reshape(B, M, 4 * D)
        scores = self.scorer(h_moves).squeeze(-1)
        return scores.masked_fill(~move_mask, float('-inf'))


# ─── Data generation ────────────────────────────────────────────

def _process_one_instance(args):
    """Generate training samples at multiple quality levels.

    For each quality level, greedy-improve a random solution by a proportional
    number of steps, then compute deltas for all feasible moves at that state.

    Returns: (samples_list, best_cost)
      samples: list of (idx, sol_copy, deltas_array, t)
    """
    idx, inst, manifold_class, max_moves, n_quality_levels, max_improve_steps = args

    if isinstance(manifold_class, type):
        manifold = manifold_class()
    else:
        manifold = manifold_class

    samples = []
    best_cost = float('inf')
    time_limit = 120
    t_start = time.time()

    for ql in range(n_quality_levels):
        if time.time() - t_start > time_limit:
            break

        sol = manifold.sample_random(inst)

        n_improve = ql * max_improve_steps // max(n_quality_levels - 1, 1)
        for _ in range(n_improve):
            if time.time() - t_start > time_limit:
                break
            moves = manifold.enumerate_moves(sol, inst)
            if not moves:
                break
            # Subsample for greedy improvement only (finding best move)
            if len(moves) > max_moves:
                subset = np.random.choice(len(moves), max_moves, replace=False)
                moves_sub = [moves[i] for i in subset]
            else:
                moves_sub = moves
            deltas_sub = np.array([manifold.move_delta(sol, m, inst) for m in moves_sub])
            best = np.argmin(deltas_sub)
            if deltas_sub[best] >= -1e-10:
                break
            sol = manifold.apply_move(sol, moves_sub[best])

        # Compute deltas for ALL moves (training uses truncation at batch time)
        moves = manifold.enumerate_moves(sol, inst)
        if not moves:
            continue

        deltas = np.array([manifold.move_delta(sol, m, inst) for m in moves],
                          dtype=np.float32)
        t = 1.0 - ql / max(n_quality_levels - 1, 1)

        samples.append((idx, sol.copy() if hasattr(sol, 'copy') else sol,
                        deltas, t))

        cost = manifold.cost(sol, inst)
        best_cost = min(best_cost, cost)

    return samples, best_cost


def build_sample_pool(config, manifold, instances, max_moves,
                      n_quality_levels=5, max_improve_steps=100,
                      n_workers=None):
    """Build training samples at multiple quality levels per instance."""
    if n_workers is None:
        n_workers = min(cpu_count(), 32)

    manifold_class = type(manifold)

    work_args = [
        (idx, inst, manifold_class, max_moves, n_quality_levels, max_improve_steps)
        for idx, inst in enumerate(instances)
    ]

    sample_pool = []
    costs = []

    if n_workers <= 1:
        work_args_serial = [
            (idx, inst, manifold, max_moves, n_quality_levels, max_improve_steps)
            for idx, inst in enumerate(instances)
        ]
        pbar = tqdm(work_args_serial, desc="  Pool generation")
        for args in pbar:
            samples, cost = _process_one_instance(args)
            sample_pool.extend(samples)
            costs.append(cost)
            pbar.set_postfix(samples=len(sample_pool), cost=f"{cost:.2f}")
    else:
        print(f"  Pool generation: {len(instances)} instances × {n_quality_levels} levels, "
              f"{n_workers} workers")
        with mp.Pool(n_workers) as p:
            for samples, cost in tqdm(
                p.imap_unordered(_process_one_instance, work_args, chunksize=1),
                total=len(instances),
                desc="  Pool generation",
            ):
                sample_pool.extend(samples)
                costs.append(cost)

    return sample_pool, costs


def calibrate_tau(manifold, instances, max_moves, n_samples=10):
    """Estimate typical delta magnitude for temperature calibration."""
    all_deltas = []
    for inst in instances[:n_samples]:
        sol = manifold.sample_random(inst)
        moves = manifold.enumerate_moves(sol, inst)
        for m in moves[:50]:
            all_deltas.append(abs(manifold.move_delta(sol, m, inst)))
    delta_scale = np.median(all_deltas) if all_deltas else 0.1
    return delta_scale * 0.5, delta_scale * 5.0


def boltzmann_target(deltas, tau):
    """Soft target: softmax(-delta / tau). Favor improving moves."""
    logits = -deltas / max(tau, 1e-8)
    logits = logits - logits.max()
    probs = np.exp(logits)
    return (probs / probs.sum()).astype(np.float32)


def prepare_batch_item(config, solution, instance, moves, best_idx, t, max_moves):
    """Prepare tensors for one training sample."""
    node_features = config.build_node_features(solution, instance, t)
    edge_index = config.build_edges(solution, instance)

    move_nodes = np.zeros((max_moves, 4), dtype=np.int64)
    move_mask = np.zeros(max_moves, dtype=bool)

    for i, m in enumerate(moves[:max_moves]):
        a, b, c, d = config.move_to_4nodes(solution, m, instance)
        move_nodes[i] = [a, b, c, d]
        move_mask[i] = True

    label = min(best_idx, max_moves - 1)
    return node_features, edge_index, move_nodes, move_mask, label


# ─── Inference ──────────────────────────────────────────────────

@torch.no_grad()
def greedy_denoise(model, config, manifold, instance, max_moves, n_steps, device):
    """Greedy denoising: argmax scores, apply if delta < 0."""
    model.eval()
    sol = manifold.sample_random(instance)

    for step in range(n_steps):
        moves = manifold.enumerate_moves(sol, instance)
        if not moves:
            break
        if len(moves) > max_moves:
            subset = np.random.choice(len(moves), max_moves, replace=False)
            moves = [moves[i] for i in subset]

        t_val = 1.0 - step / max(n_steps - 1, 1)
        nf, ei, mn, mm, _ = prepare_batch_item(
            config, sol, instance, moves, 0, t_val, max_moves
        )

        nf_t = torch.tensor(nf, dtype=torch.float32, device=device).unsqueeze(0)
        ei_t = torch.tensor(ei, dtype=torch.long, device=device).unsqueeze(0)
        mn_t = torch.tensor(mn, dtype=torch.long, device=device).unsqueeze(0)
        mm_t = torch.tensor(mm, dtype=torch.bool, device=device).unsqueeze(0)
        t_t = torch.tensor([t_val], dtype=torch.float32, device=device)

        scores = model(nf_t, ei_t, mn_t, mm_t, t_t)
        best = scores.argmax(dim=-1).item()

        if best < len(moves):
            d = manifold.move_delta(sol, moves[best], instance)
            if d < 0:
                sol = manifold.apply_move(sol, moves[best])

    cost = manifold.cost(sol, instance)
    return sol, cost


@torch.no_grad()
def stochastic_denoise(model, config, manifold, instance, max_moves, n_steps,
                       device, temperature=0.5):
    """Stochastic denoising: sample from score distribution."""
    model.eval()
    sol = manifold.sample_random(instance)

    for step in range(n_steps):
        moves = manifold.enumerate_moves(sol, instance)
        if not moves:
            break
        if len(moves) > max_moves:
            subset = np.random.choice(len(moves), max_moves, replace=False)
            moves = [moves[i] for i in subset]

        t_val = 1.0 - step / max(n_steps - 1, 1)
        temp = max(temperature * t_val, 0.05)

        nf, ei, mn, mm, _ = prepare_batch_item(
            config, sol, instance, moves, 0, t_val, max_moves
        )

        nf_t = torch.tensor(nf, dtype=torch.float32, device=device).unsqueeze(0)
        ei_t = torch.tensor(ei, dtype=torch.long, device=device).unsqueeze(0)
        mn_t = torch.tensor(mn, dtype=torch.long, device=device).unsqueeze(0)
        mm_t = torch.tensor(mm, dtype=torch.bool, device=device).unsqueeze(0)
        t_t = torch.tensor([t_val], dtype=torch.float32, device=device)

        scores = model(nf_t, ei_t, mn_t, mm_t, t_t)
        probs = F.softmax(scores.squeeze(0)[:len(moves)] / temp, dim=-1)
        best = torch.multinomial(probs, 1).item()

        if best < len(moves):
            d = manifold.move_delta(sol, moves[best], instance)
            if d < 0:
                sol = manifold.apply_move(sol, moves[best])

    cost = manifold.cost(sol, instance)
    return sol, cost


@torch.no_grad()
def best_of_k_denoise(model, config, manifold, instance, max_moves, n_steps,
                      device, K=8, temperature=0.5):
    """Run K stochastic samples, return the best."""
    best_sol, best_cost = None, float('inf')
    for _ in range(K):
        sol, cost = stochastic_denoise(
            model, config, manifold, instance, max_moves, n_steps,
            device, temperature
        )
        if cost < best_cost:
            best_cost = cost
            best_sol = sol
    return best_sol, best_cost


# ─── Training loop ──────────────────────────────────────────────

def train(args):
    device = torch.device(args.device)

    ConfigClass = PROBLEM_CONFIGS[args.problem]
    config = ConfigClass()
    manifold = config.create_manifold()
    print(f"Problem: {config.name}, N={args.N}")

    # Generate instances
    print(f"Generating {args.n_instances} training instances...")
    instances = [config.create_instance(args.N, seed=42 + i) for i in range(args.n_instances)]
    val_instances = [config.create_instance(args.N, seed=99999 + i) for i in range(args.n_val)]

    # Auto-calibrate temperature
    print("Calibrating temperature range...")
    tau_min, tau_max = calibrate_tau(manifold, instances, args.max_moves)
    print(f"  tau_min={tau_min:.4f}, tau_max={tau_max:.4f}")

    # Build training pool (before GPU model to avoid CUDA fork deadlock)
    print("Building training sample pool...")
    t0 = time.time()
    pool, train_costs = build_sample_pool(
        config, manifold, instances, args.max_moves,
        n_quality_levels=args.n_quality_levels,
    )
    print(f"  {len(pool)} samples in {time.time() - t0:.1f}s, "
          f"avg cost: {np.mean(train_costs):.4f}")

    _, val_costs = build_sample_pool(
        config, manifold, val_instances, args.max_moves,
        n_quality_levels=args.n_quality_levels,
    )

    # Model (create after pool generation)
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
            batch_nf, batch_ei, batch_mn, batch_mm = [], [], [], []
            batch_targets, batch_t = [], []
            max_edges = 0

            items = []
            for _ in range(args.batch_size):
                s_idx = np.random.randint(n_pool)
                idx, sol, deltas, t = pool[s_idx]

                moves = manifold.enumerate_moves(sol, instances[idx])
                if len(moves) != len(deltas) or not moves:
                    continue

                # Truncate to max_moves consistently
                n_use = min(len(moves), args.max_moves)
                tau_t = tau_min + t * (tau_max - tau_min)
                target = boltzmann_target(deltas[:n_use], tau_t)

                nf, ei, mn, mm, _ = prepare_batch_item(
                    config, sol, instances[idx], moves[:n_use], 0, t, args.max_moves
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

            # Track accuracy: does model's top pick match Boltzmann's top pick?
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
        print(f"Epoch {epoch}: loss={epoch_loss / args.steps_per_epoch:.4f}, "
              f"acc={epoch_acc / args.steps_per_epoch:.2%}")

        if epoch % args.eval_freq == 0:
            eval_costs = []
            for vi in range(min(args.n_eval, len(val_instances))):
                _, cost = greedy_denoise(
                    model, config, manifold, val_instances[vi],
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
                    'tau_min': tau_min, 'tau_max': tau_max,
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
    parser.add_argument('--n_quality_levels', type=int, default=5)
    parser.add_argument('--n_denoise', type=int, default=None)
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--n_eval', type=int, default=20)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints')
    args = parser.parse_args()
    train(args)
