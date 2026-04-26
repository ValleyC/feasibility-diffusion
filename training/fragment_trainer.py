"""
Fragment Merge Trainer: learns to score oriented fragment merges for CVRPTW.

Training pipeline:
  1. Initialize elite buffer with savings heuristic solutions
  2. Extract merge targets from elite consensus (adjacency + co-route frequency)
  3. Train GNN to predict merge scores and infeasibility risk
  4. Self-improvement: decode with current model, add better solutions to buffer
  5. Repeat

Losses:
  - L_merge: BCE on merge scores vs elite adjacency frequency
  - L_risk: BCE on risk scores vs actual infeasibility (from exact TW check)
  - L_cost: ranking loss — decoded cost with learned scores vs savings baseline

Usage:
    python -m training.fragment_trainer --n_customers 50 --device cuda:0
"""

import sys, os, argparse, time, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from data.cvrptw_generator import generate_dataset
from route_objects.fragment import (
    RouteFragment, create_singleton, check_merge_feasible,
    simulate_route, route_cost_with_depot,
)
from route_objects.fragment_state import FragmentState, build_fragment_graph
from route_objects.projector import (
    project, project_savings, project_random, finalize_routes,
    local_search_repair,
)
from models.fragment_gnn import FragmentGNN
from training.elite_buffer import EliteBuffer, extract_merge_targets


def build_training_batch(buffer: EliteBuffer, instances: list,
                         batch_indices: list, k: int = 20):
    """Build a training batch from elite solutions.

    For each instance in the batch:
      1. Start from singletons (or a random partial merge state)
      2. Build fragment graph
      3. Label each edge using elite consensus

    Returns lists of (node_feat, edge_index, edge_feat, merge_labels, risk_labels)
    """
    batch = []

    for inst_idx in batch_indices:
        inst = instances[inst_idx]
        adjacency_freq, coroute_freq = extract_merge_targets(buffer, inst_idx)

        # Start from singletons
        state = FragmentState.from_singletons(inst)
        node_feat, edge_index, edge_feat = build_fragment_graph(state, k=k)

        if len(edge_index) == 0:
            continue

        # Compute per-edge labels
        E = len(edge_index)
        merge_labels = np.zeros(E, dtype=np.float32)
        risk_labels = np.zeros(E, dtype=np.float32)

        for e in range(E):
            src_idx = edge_index[e, 0]
            dst_idx = edge_index[e, 1]
            f_src = state.fragments[src_idx]
            f_dst = state.fragments[dst_idx]

            # Merge label: are src.end_node and dst.start_node adjacent in elites?
            merge_labels[e] = adjacency_freq[f_src.end_node, f_dst.start_node]

            # Risk label: is this merge actually infeasible?
            feasible, _ = check_merge_feasible(f_src, f_dst, inst, orientation=0)
            risk_labels[e] = 0.0 if feasible else 1.0

        batch.append({
            'node_feat': node_feat,
            'edge_index': edge_index,
            'edge_feat': edge_feat,
            'merge_labels': merge_labels,
            'risk_labels': risk_labels,
        })

    return batch


def train_step(model, optimizer, batch, device):
    """One training step on a batch of fragment graphs."""
    model.train()
    total_loss = 0.0
    total_l_merge = 0.0
    total_l_risk = 0.0
    n_items = 0

    for item in batch:
        nf = torch.tensor(item['node_feat'], dtype=torch.float32, device=device)
        ei = torch.tensor(item['edge_index'], dtype=torch.long, device=device)
        ef = torch.tensor(item['edge_feat'], dtype=torch.float32, device=device)
        ml = torch.tensor(item['merge_labels'], dtype=torch.float32, device=device)
        rl = torch.tensor(item['risk_labels'], dtype=torch.float32, device=device)

        if ei.shape[0] == 0:
            continue

        merge_scores, risk_scores = model(nf, ei, ef)

        # L_merge: BCE on merge scores vs elite frequency
        l_merge = F.binary_cross_entropy_with_logits(merge_scores, ml)

        # L_risk: BCE on risk scores vs actual infeasibility
        l_risk = F.binary_cross_entropy(risk_scores, rl)

        loss = l_merge + 0.5 * l_risk
        total_loss += loss.item()
        total_l_merge += l_merge.item()
        total_l_risk += l_risk.item()
        n_items += 1

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    if n_items == 0:
        return 0.0, 0.0, 0.0
    return total_loss / n_items, total_l_merge / n_items, total_l_risk / n_items


@torch.no_grad()
def evaluate(model, instances, device, n_eval=20, k=20, repair_steps=50):
    """Evaluate model on validation instances.

    Returns:
        costs_model: list of costs with learned merges
        costs_savings: list of costs with savings baseline
        costs_random: list of costs with random merges
        diagnostics: dict of pre-repair metrics
    """
    model.eval()
    costs_model, costs_savings, costs_random = [], [], []
    pre_repair_costs = []
    feasible_merge_rates = []

    for vi in range(min(n_eval, len(instances))):
        inst = instances[vi]

        # Model-based
        state_m = FragmentState.from_singletons(inst)
        state_m = project(model, state_m, device, k=k, max_steps=500)
        pre_repair_costs.append(state_m.total_cost())
        state_m = local_search_repair(state_m, max_steps=repair_steps)
        costs_model.append(state_m.total_cost())

        # Savings baseline
        state_s = FragmentState.from_singletons(inst)
        state_s = project_savings(state_s, max_steps=500)
        state_s = local_search_repair(state_s, max_steps=repair_steps)
        costs_savings.append(state_s.total_cost())

        # Random baseline
        state_r = FragmentState.from_singletons(inst)
        state_r = project_random(state_r, max_steps=500)
        state_r = local_search_repair(state_r, max_steps=repair_steps)
        costs_random.append(state_r.total_cost())

    diagnostics = {
        'pre_repair_cost': np.mean(pre_repair_costs),
        'post_repair_cost': np.mean(costs_model),
        'savings_cost': np.mean(costs_savings),
        'random_cost': np.mean(costs_random),
    }
    return costs_model, costs_savings, costs_random, diagnostics


def train(args):
    device = torch.device(args.device)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print(f"Generating {args.n_instances} CVRPTW-{args.n_customers} instances...")
    train_instances = generate_dataset(
        args.n_customers, args.n_instances, seed=42,
        capacity_ratio=args.capacity_ratio, tw_width=args.tw_width)
    val_instances = generate_dataset(
        args.n_customers, args.n_val, seed=99999,
        capacity_ratio=args.capacity_ratio, tw_width=args.tw_width)

    print(f"Building elite buffer with savings heuristic...")
    buffer = EliteBuffer(train_instances, n_elite=args.n_elite)
    buffer.initialize_with_savings(n_restarts=args.n_elite)
    avg_elite = np.mean([buffer.best_cost(i) for i in range(len(train_instances))])
    print(f"  Avg elite cost: {avg_elite:.4f}")

    # Validation savings baseline
    val_savings = []
    for inst in val_instances[:args.n_eval]:
        s = FragmentState.from_singletons(inst)
        s = project_savings(s, max_steps=500)
        s = local_search_repair(s, max_steps=50)
        val_savings.append(s.total_cost())
    ref_cost = np.mean(val_savings)
    print(f"  Val savings baseline: {ref_cost:.4f}")

    # Model
    model = FragmentGNN(
        n_node_feat=12, n_edge_feat=3,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers,
    ).to(device)
    print(f"  FragmentGNN: {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs)

    best_gap = float('inf')

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        epoch_loss, epoch_merge, epoch_risk = 0.0, 0.0, 0.0
        n_steps = 0

        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch}")
        for step in pbar:
            # Sample batch of instances
            batch_idx = np.random.choice(
                len(train_instances), args.batch_size, replace=False)
            batch = build_training_batch(
                buffer, train_instances, batch_idx, k=args.k)

            if not batch:
                continue

            loss, l_m, l_r = train_step(model, optimizer, batch, device)
            epoch_loss += loss
            epoch_merge += l_m
            epoch_risk += l_r
            n_steps += 1
            pbar.set_postfix(loss=f"{loss:.3f}", merge=f"{l_m:.3f}", risk=f"{l_r:.3f}")

        scheduler.step()
        if n_steps > 0:
            print(f"  Epoch {epoch}: loss={epoch_loss/n_steps:.4f} "
                  f"[merge={epoch_merge/n_steps:.3f} risk={epoch_risk/n_steps:.3f}]")

        # Evaluate
        if epoch % args.eval_freq == 0 or epoch == args.n_epochs:
            costs_m, costs_s, costs_r, diag = evaluate(
                model, val_instances, device, n_eval=args.n_eval, k=args.k)

            avg_m = np.mean(costs_m)
            avg_s = np.mean(costs_s)
            avg_r = np.mean(costs_r)
            gap_vs_savings = (avg_m / avg_s - 1) * 100

            print(f"  Val model:   {avg_m:.4f} ({gap_vs_savings:+.2f}% vs savings)")
            print(f"  Val savings: {avg_s:.4f}")
            print(f"  Val random:  {avg_r:.4f}")
            print(f"  Pre-repair:  {diag['pre_repair_cost']:.4f}")

            if gap_vs_savings < best_gap:
                best_gap = gap_vs_savings
                path = os.path.join(args.ckpt_dir, 'best.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'gap': gap_vs_savings,
                    'args': vars(args),
                }, path)
                print(f"  [best] gap={gap_vs_savings:+.2f}% -> {path}")

        # Self-improvement
        if epoch % args.self_improve_freq == 0:
            n_improved = 0
            for i in range(len(train_instances)):
                state = FragmentState.from_singletons(train_instances[i])
                state = project(model, state, device, k=args.k, max_steps=500)
                state = local_search_repair(state, max_steps=50)
                routes = [f.seq for f in state.fragments]
                cost = state.total_cost()
                if buffer.add(i, routes, cost):
                    n_improved += 1
            print(f"  Self-improvement: {n_improved}/{len(train_instances)} "
                  f"instances improved")

    print(f"\nDone. Best gap vs savings: {best_gap:+.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fragment Merge Trainer')
    parser.add_argument('--n_customers', type=int, default=50)
    parser.add_argument('--n_instances', type=int, default=500)
    parser.add_argument('--n_val', type=int, default=50)
    parser.add_argument('--n_elite', type=int, default=5)
    parser.add_argument('--n_eval', type=int, default=20)
    parser.add_argument('--capacity_ratio', type=float, default=0.8)
    parser.add_argument('--tw_width', type=str, default='medium')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--steps_per_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--self_improve_freq', type=int, default=10)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/fragment')
    args = parser.parse_args()
    train(args)
