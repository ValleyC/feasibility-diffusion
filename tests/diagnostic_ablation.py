"""
Diagnostic ablation: isolate the cause of policy drift.

Three runs, identical except for one factor:
  A: Frozen buffer, no RL, no self-improvement (pure supervised)
  B: Frozen buffer, with RL, no self-improvement (supervised + RL)
  C: Frozen buffer, no RL, with self-improvement (supervised + drift test)

Key metric: pre-repair validation cost over epochs.
If A degrades → supervised target is the problem (adjacency-label mismatch)
If A is stable but B degrades → RL is destabilizing
If A is stable but C degrades → self-improvement is contaminating
"""

import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from tqdm import tqdm

from data.cvrptw_generator import generate_dataset
from route_objects.fragment_state import FragmentState, build_fragment_graph
from route_objects.fragment import check_merge_feasible
from route_objects.projector import (
    project, project_savings, project_random, local_search_repair,
)
from models.fragment_gnn import FragmentGNN
from training.elite_buffer import EliteBuffer, extract_merge_targets
from training.fragment_trainer import (
    build_training_batch, train_step, cost_improvement_step,
)


def evaluate_detailed(model, instances, device, k=15, n_eval=10):
    """Detailed evaluation with pre-repair diagnostics."""
    model.eval()
    pre_costs, post_costs, n_frags_list = [], [], []

    for vi in range(min(n_eval, len(instances))):
        inst = instances[vi]

        state = FragmentState.from_singletons(inst)
        state = project(model, state, device, k=k, max_steps=500)
        pre_costs.append(state.total_cost())
        n_frags_list.append(state.n_fragments)

        state = local_search_repair(state, max_steps=30)
        post_costs.append(state.total_cost())

    return {
        'pre_repair': np.mean(pre_costs),
        'post_repair': np.mean(post_costs),
        'n_fragments': np.mean(n_frags_list),
    }


def run_ablation(name, train_instances, val_instances, val_savings_cost,
                 device, n_epochs=20, use_rl=False, use_self_improve=False):
    """Run one ablation variant."""
    print(f"\n{'='*60}")
    print(f"  ABLATION {name}")
    print(f"  RL={use_rl}, self-improve={use_self_improve}")
    print(f"{'='*60}")

    buffer = EliteBuffer(train_instances, n_elite=5)
    buffer.initialize_with_savings(n_restarts=3)
    avg_elite = np.mean([buffer.best_cost(i) for i in range(len(train_instances))])
    print(f"  Elite avg: {avg_elite:.4f}")

    model = FragmentGNN(n_node_feat=12, n_edge_feat=3,
                        hidden_dim=96, n_layers=4).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    results = []

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_steps = 0

        for step in range(50):
            batch_idx = np.random.choice(len(train_instances), 8, replace=False)
            batch = build_training_batch(buffer, train_instances, batch_idx, k=15)
            if not batch:
                continue
            loss, l_m, l_r, _ = train_step(
                model, optimizer, batch, device, train_instances)
            epoch_loss += loss
            n_steps += 1

        # RL cost step (if enabled)
        cost_rl = 0.0
        if use_rl and epoch >= 3:
            for _ in range(3):
                cost_rl += cost_improvement_step(
                    model, optimizer, train_instances, device,
                    n_instances=4, k=15, lambda_cost=0.05)
            cost_rl /= 3

        # Self-improvement (if enabled)
        n_improved = 0
        if use_self_improve and epoch % 5 == 0:
            for i in range(len(train_instances)):
                state = FragmentState.from_singletons(train_instances[i])
                state = project(model, state, device, k=15, max_steps=500)
                state = local_search_repair(state, max_steps=30)
                routes = [f.seq for f in state.fragments]
                cost = state.total_cost()
                if buffer.add(i, routes, cost):
                    n_improved += 1

        # Evaluate every 2 epochs
        if epoch % 2 == 0 or epoch == 1:
            diag = evaluate_detailed(model, val_instances, device, k=15, n_eval=10)
            gap = (diag['post_repair'] / val_savings_cost - 1) * 100

            results.append({
                'epoch': epoch,
                'train_loss': epoch_loss / max(n_steps, 1),
                'pre_repair': diag['pre_repair'],
                'post_repair': diag['post_repair'],
                'gap': gap,
                'n_fragments': diag['n_fragments'],
                'cost_rl': cost_rl,
                'n_improved': n_improved,
            })

            status = f"  Ep {epoch:2d}: loss={epoch_loss/max(n_steps,1):.4f} " \
                     f"pre={diag['pre_repair']:.2f} post={diag['post_repair']:.2f} " \
                     f"gap={gap:+.2f}% frags={diag['n_fragments']:.1f}"
            if use_rl:
                status += f" rl={cost_rl:.3f}"
            if n_improved > 0:
                status += f" improved={n_improved}/{len(train_instances)}"
            print(status)

    return results


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    N = 30  # moderate size for meaningful but fast test
    n_train = 200
    n_val = 20

    print(f"Generating CVRPTW-{N} instances: {n_train} train, {n_val} val")
    train_instances = generate_dataset(N, n_train, seed=42,
                                       capacity_ratio=0.8, tw_width='medium')
    val_instances = generate_dataset(N, n_val, seed=99999,
                                     capacity_ratio=0.8, tw_width='medium')

    # Compute savings baseline
    val_savings = []
    for inst in val_instances[:10]:
        s = FragmentState.from_singletons(inst)
        s = project_savings(s, max_steps=500)
        s = local_search_repair(s, max_steps=30)
        val_savings.append(s.total_cost())
    val_savings_cost = np.mean(val_savings)
    print(f"Val savings baseline: {val_savings_cost:.4f}")

    # Run three ablations
    results_a = run_ablation("A: Pure Supervised (frozen buffer, no RL)",
                             train_instances, val_instances, val_savings_cost,
                             device, n_epochs=20, use_rl=False, use_self_improve=False)

    results_b = run_ablation("B: Supervised + RL (frozen buffer)",
                             train_instances, val_instances, val_savings_cost,
                             device, n_epochs=20, use_rl=True, use_self_improve=False)

    results_c = run_ablation("C: Supervised + Self-Improve (no RL)",
                             train_instances, val_instances, val_savings_cost,
                             device, n_epochs=20, use_rl=False, use_self_improve=True)

    # Summary
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Savings baseline: {val_savings_cost:.4f}")
    print()
    print(f"  {'Epoch':>5} | {'A: Pure Sup':>20} | {'B: Sup+RL':>20} | {'C: Sup+SI':>20}")
    print(f"  {'':>5} | {'pre / post / gap':>20} | {'pre / post / gap':>20} | {'pre / post / gap':>20}")
    print(f"  {'-'*5}-+-{'-'*20}-+-{'-'*20}-+-{'-'*20}")

    # Align by epoch
    epochs_a = {r['epoch']: r for r in results_a}
    epochs_b = {r['epoch']: r for r in results_b}
    epochs_c = {r['epoch']: r for r in results_c}
    all_epochs = sorted(set(list(epochs_a.keys()) + list(epochs_b.keys()) + list(epochs_c.keys())))

    for ep in all_epochs:
        parts = []
        for d in [epochs_a, epochs_b, epochs_c]:
            if ep in d:
                r = d[ep]
                parts.append(f"{r['pre_repair']:6.2f}/{r['post_repair']:5.2f}/{r['gap']:+5.1f}%")
            else:
                parts.append(f"{'---':>20}")
        print(f"  {ep:5d} | {parts[0]:>20} | {parts[1]:>20} | {parts[2]:>20}")

    # Diagnosis
    print(f"\n  DIAGNOSIS:")
    a_pre_trend = [r['pre_repair'] for r in results_a]
    b_pre_trend = [r['pre_repair'] for r in results_b]
    c_pre_trend = [r['pre_repair'] for r in results_c]

    a_drift = a_pre_trend[-1] - a_pre_trend[0] if len(a_pre_trend) > 1 else 0
    b_drift = b_pre_trend[-1] - b_pre_trend[0] if len(b_pre_trend) > 1 else 0
    c_drift = c_pre_trend[-1] - c_pre_trend[0] if len(c_pre_trend) > 1 else 0

    print(f"  A (Pure Supervised) pre-repair drift: {a_drift:+.2f}")
    print(f"  B (Sup + RL)        pre-repair drift: {b_drift:+.2f}")
    print(f"  C (Sup + Self-Imp)  pre-repair drift: {c_drift:+.2f}")

    if a_drift > 2.0:
        print(f"  → Supervised target itself is the problem (adjacency-label mismatch)")
    elif b_drift > a_drift + 2.0:
        print(f"  → RL is destabilizing the policy")
    elif c_drift > a_drift + 2.0:
        print(f"  → Self-improvement is contaminating the buffer")
    elif a_drift < 1.0 and b_drift < 1.0 and c_drift < 1.0:
        print(f"  → All variants are stable — problem may be scale/capacity dependent")
    else:
        print(f"  → Mixed signal — multiple factors may be interacting")

    # Best gap for each
    for name, results in [("A", results_a), ("B", results_b), ("C", results_c)]:
        best = min(results, key=lambda r: r['gap'])
        print(f"  {name} best: epoch {best['epoch']}, gap={best['gap']:+.2f}%, "
              f"pre={best['pre_repair']:.2f}")


if __name__ == '__main__':
    main()
