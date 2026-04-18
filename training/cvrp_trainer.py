"""
Clean label-free training for feasibility-preserving diffusion on CVRP.

Same principle as TSP: greedy local search generates its own training data.
Each improvement step yields (solution, best_move) covering the full quality
spectrum. The model learns which feasibility-preserving move to apply.

Three move types (all preserve capacity constraints):
  1. Intra-route 2-opt (reverse segment within a route)
  2. Inter-route relocate (move customer between routes, capacity-checked)
  3. Inter-route swap (exchange customers, capacity-checked)

Every intermediate state is a valid CVRP solution.

Usage:
    python -m training.cvrp_trainer --N 20 --device cuda:0
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from problems.cvrp.solution import (
    random_solution, is_feasible, solution_cost,
    enumerate_moves, apply_move, delta_move,
)
from problems.cvrp.data import (
    generate_cvrp_instance, solution_to_edges, solution_to_features,
    move_to_4nodes, greedy_improve,
)
from problems.tsp.tour import dist_matrix_from_coords, random_tour


# ── Model ────────────────────────────────────────────────────

class CVRPGNNLayer(nn.Module):
    """Message-passing on the route graph (edges = consecutive nodes in routes)."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.msg_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h, edge_index):
        """
        h: (B, N+1, D) node features
        edge_index: (B, E, 2) edge list (source, target)
        """
        B, N1, D = h.shape
        E = edge_index.shape[1]

        # Gather source and target features
        src = edge_index[:, :, 0]  # (B, E)
        tgt = edge_index[:, :, 1]  # (B, E)

        h_src = h.gather(1, src.unsqueeze(-1).expand(-1, -1, D))  # (B, E, D)
        h_tgt = h.gather(1, tgt.unsqueeze(-1).expand(-1, -1, D))  # (B, E, D)

        # Messages
        msg = self.msg_mlp(torch.cat([h_tgt, h_src], dim=-1))  # (B, E, D)

        # Aggregate: sum messages per target node
        agg = torch.zeros_like(h)
        agg.scatter_add_(1, tgt.unsqueeze(-1).expand(-1, -1, D), msg)

        # Update
        h_new = self.update_mlp(torch.cat([h, agg], dim=-1))
        return self.norm(h + h_new)


class CVRPMoveScorer(nn.Module):
    """Score candidate CVRP moves using GNN on the route graph.

    Architecture:
      1. Node features: (x, y, demand/cap, route_id/K, position, is_depot, time)
      2. GNN on route graph (edges = consecutive nodes in routes)
      3. For each move: extract 4 critical node embeddings → MLP → score
    """

    def __init__(self, n_node_features=7, hidden_dim=64, n_layers=4):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.node_embed = nn.Sequential(
            nn.Linear(n_node_features, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gnn_layers = nn.ModuleList([
            CVRPGNNLayer(hidden_dim) for _ in range(n_layers)
        ])
        self.move_scorer = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, node_features, edge_index, move_nodes, move_mask):
        """
        Args:
            node_features: (B, N+1, F) per-node features
            edge_index: (B, E, 2) route-graph edges
            move_nodes: (B, M, 4) long — 4 critical node indices per move
            move_mask: (B, M) bool — True for valid moves, False for padding

        Returns:
            scores: (B, M) logits per move (masked positions = -inf)
        """
        B = node_features.shape[0]
        D = self.hidden_dim

        # Encode nodes
        h = self.node_embed(node_features)

        # GNN message passing
        for layer in self.gnn_layers:
            h = layer(h, edge_index)

        # Score each move from 4 node embeddings
        M = move_nodes.shape[1]
        # Gather: (B, M, 4, D)
        idx = move_nodes.unsqueeze(-1).expand(-1, -1, -1, D)  # (B, M, 4, D)
        h_moves = h.unsqueeze(1).expand(-1, M, -1, -1).gather(2, idx)  # (B, M, 4, D)
        h_moves = h_moves.reshape(B, M, 4 * D)

        scores = self.move_scorer(h_moves).squeeze(-1)  # (B, M)
        scores = scores.masked_fill(~move_mask, float('-inf'))

        return scores


# ── Data preparation ─────────────────────────────────────────

def prepare_batch_item(routes, instance, moves_list, progress, max_moves):
    """Prepare tensors for a single training sample.

    Returns:
        node_features: (N+1, F) float
        edge_index: (E, 2) long
        move_nodes: (max_moves, 4) long (padded)
        move_mask: (max_moves,) bool
        label: int (index of best-improving move)
    """
    N = instance['n_customers']
    coords = instance['coords']
    demands = instance['demands']
    capacity = instance['capacity']

    # Node features: x, y, demand/cap, route_id/K, position, is_depot, progress
    feat_info = solution_to_features(routes, N, demands, capacity)
    K = len(routes)
    K = max(K, 1)

    node_features = np.zeros((N + 1, 7), dtype=np.float32)
    node_features[:, 0:2] = coords
    node_features[:, 2] = demands / capacity
    node_features[:, 3] = feat_info['route_ids'] / K
    node_features[:, 4] = feat_info['positions']
    node_features[:, 5] = feat_info['is_depot']
    node_features[:, 6] = progress

    # Edges from route structure (bidirectional)
    edges = solution_to_edges(routes)
    edge_list = []
    for a, b in edges:
        edge_list.append([a, b])
        edge_list.append([b, a])
    edge_index = np.array(edge_list, dtype=np.int64) if edge_list else np.zeros((0, 2), dtype=np.int64)

    # Move node indices + mask
    move_nodes_list = [move_to_4nodes(routes, m) for m in moves_list]
    n_moves = len(moves_list)

    move_nodes = np.zeros((max_moves, 4), dtype=np.int64)
    move_mask = np.zeros(max_moves, dtype=bool)

    for i, (a, b, c, d) in enumerate(move_nodes_list[:max_moves]):
        move_nodes[i] = [a, b, c, d]
        move_mask[i] = True

    # Label: best-improving move
    dist = instance['dist']
    deltas = np.array([delta_move(routes, m, dist) for m in moves_list])
    label = int(np.argmin(deltas))

    return node_features, edge_index, move_nodes, move_mask, label


# ── Training ─────────────────────────────────────────────────

def build_sample_pool(instances, max_moves, N):
    """Build training samples from greedy improvement trajectories."""
    print(f"Building CVRP-{N} sample pool from {len(instances)} instances...")
    t0 = time.time()

    pool = []  # list of (instance_idx, routes, moves_list, best_idx, progress)
    costs = []

    for idx, inst in enumerate(instances):
        routes = random_solution(N, inst['demands'], inst['capacity'])
        _, final_cost, trajectory = greedy_improve(routes, inst, max_iters=300)
        costs.append(final_cost)

        n_steps = len(trajectory)
        for step_i, (step_routes, step_moves, step_best) in enumerate(trajectory):
            progress = step_i / max(n_steps - 1, 1)
            if len(step_moves) <= max_moves:
                pool.append((idx, step_routes, step_moves, step_best, progress))

    elapsed = time.time() - t0
    print(f"  {len(pool)} samples, avg final cost: {np.mean(costs):.4f}, "
          f"{elapsed:.1f}s")
    return pool, costs


@torch.no_grad()
def greedy_denoise(model, instance, max_moves, N, n_steps, device):
    """Greedy denoising inference for CVRP."""
    model.eval()
    routes = random_solution(N, instance['demands'], instance['capacity'])

    for step in range(n_steps):
        moves = enumerate_moves(routes, instance['demands'], instance['capacity'])
        if len(moves) == 0:
            break

        progress = step / max(n_steps - 1, 1)
        nf, ei, mn, mm, _ = prepare_batch_item(
            routes, instance, moves, progress, max_moves
        )

        nf_t = torch.tensor(nf, dtype=torch.float32, device=device).unsqueeze(0)
        ei_t = torch.tensor(ei, dtype=torch.long, device=device).unsqueeze(0)
        mn_t = torch.tensor(mn, dtype=torch.long, device=device).unsqueeze(0)
        mm_t = torch.tensor(mm, dtype=torch.bool, device=device).unsqueeze(0)

        scores = model(nf_t, ei_t, mn_t, mm_t)
        best_idx = scores.argmax(dim=-1).item()

        if best_idx < len(moves):
            d = delta_move(routes, moves[best_idx], instance['dist'])
            if d < 0:
                routes = apply_move(routes, moves[best_idx])

    cost = solution_cost(routes, instance['dist'])
    return routes, cost


def train(args):
    device = torch.device(args.device)
    N = args.N
    max_moves = args.max_moves

    print(f"CVRP-{N}, max_moves={max_moves}")

    # Generate instances
    print(f"Generating {args.n_instances} instances...")
    instances = [generate_cvrp_instance(N, seed=42 + i) for i in range(args.n_instances)]
    val_instances = [generate_cvrp_instance(N, seed=99999 + i) for i in range(args.n_val)]

    # Build training pool
    pool, train_costs = build_sample_pool(instances, max_moves, N)

    # Validation reference
    _, val_costs = build_sample_pool(val_instances, max_moves, N)

    # Model
    model = CVRPMoveScorer(
        n_node_features=7,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"CVRPMoveScorer: {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    print(f"Baselines: greedy-2opt={np.mean(train_costs):.4f}")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    best_gap = float('inf')
    n_denoise = args.n_denoise if args.n_denoise else N * 3

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        epoch_loss, epoch_acc = 0.0, 0.0

        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch}")
        for step in pbar:
            # Sample batch
            batch_nf, batch_ei, batch_mn, batch_mm, batch_labels = [], [], [], [], []
            max_edges = 0

            items = []
            for _ in range(args.batch_size):
                s_idx = np.random.randint(len(pool))
                inst_idx, routes, moves, best_idx, progress = pool[s_idx]
                nf, ei, mn, mm, label = prepare_batch_item(
                    routes, instances[inst_idx], moves, progress, max_moves
                )
                items.append((nf, ei, mn, mm, label))
                max_edges = max(max_edges, ei.shape[0])

            # Pad edges to same length and stack
            for nf, ei, mn, mm, label in items:
                # Pad edge_index
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
        print(f"Epoch {epoch}: loss={epoch_loss/args.steps_per_epoch:.4f}, "
              f"acc={epoch_acc/args.steps_per_epoch:.2%}")

        if epoch % args.eval_freq == 0:
            eval_costs = []
            for idx in range(min(args.n_eval, len(val_instances))):
                _, cost = greedy_denoise(
                    model, val_instances[idx], max_moves, N, n_denoise, device
                )
                assert is_feasible(
                    random_solution(N, val_instances[idx]['demands'],
                                    val_instances[idx]['capacity']),
                    val_instances[idx]['demands'],
                    val_instances[idx]['capacity'], N
                )
                eval_costs.append(cost)

            avg_cost = np.mean(eval_costs)
            ref = np.mean(val_costs[:len(eval_costs)])
            gap = (avg_cost / ref - 1) * 100
            print(f"  Eval: cost={avg_cost:.4f}, gap={gap:.2f}% "
                  f"(ref={ref:.4f})")

            if gap < best_gap:
                best_gap = gap
                path = os.path.join(args.ckpt_dir, f'best_cvrp{N}.pt')
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'gap': gap, 'cost': avg_cost, 'args': vars(args),
                }, path)
                print(f"  [checkpoint] best gap={gap:.2f}% -> {path}")

    print(f"\nTraining complete. Best gap: {best_gap:.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CVRP feasibility-preserving training')
    parser.add_argument('--N', type=int, default=20)
    parser.add_argument('--n_instances', type=int, default=2000)
    parser.add_argument('--n_val', type=int, default=50)
    parser.add_argument('--max_moves', type=int, default=500,
                        help='Max moves to score per solution (pad/truncate)')
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
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/cvrp')
    args = parser.parse_args()
    train(args)
