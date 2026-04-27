"""
Decisive ablation: Plain GNN vs Constraint-Native GNN.

Same decoder. Same training (REINFORCE). Same compute budget.
Only the encoder layer changes.

Evaluate on:
  1. Standard CVRP (fixed capacity)
  2. Constraint shift: train on cap_ratio=0.8, test on 0.5 / 0.6 / 1.0 / 1.2
  3. Pre-repair diagnostics: does constraint-native produce better raw merges?

If the constraint-native layer wins on constraint shifts with matched
parameters and fixed decoder, the architectural contribution is real.
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.cvrp_generator import generate_cvrp_dataset, generate_cvrp_instance
from route_objects.fragment import check_merge_feasible, simulate_route
from route_objects.fragment_state import FragmentState, build_fragment_graph
from route_objects.projector import (
    project, project_savings, local_search_repair,
)
from models.fragment_gnn import FragmentGNN
from models.constraint_native import ConstraintNativeGNN
from training.rl_trainer import rollout_stochastic, rollout_greedy


def train_one_epoch(model, optimizer, train_instances, device, k,
                    steps=50, batch_size=4, temperature=2.0):
    """One epoch of REINFORCE training."""
    model.train()
    total_reward = 0.0
    n = 0

    for step in range(steps):
        batch_log_probs = []
        batch_rewards = []
        batch_baselines = []

        for _ in range(batch_size):
            idx = np.random.randint(len(train_instances))
            inst = train_instances[idx]

            state_s, lps = rollout_stochastic(
                model, inst, device, k=k, temperature=temperature,
                max_steps=inst['n_customers'] * 2)
            if not lps:
                continue
            cost_s = state_s.total_cost()

            state_g = rollout_greedy(model, inst, device, k=k,
                                     max_steps=inst['n_customers'] * 2)
            cost_g = state_g.total_cost()

            batch_rewards.append(-cost_s)
            batch_baselines.append(-cost_g)
            batch_log_probs.append(torch.stack(lps))

        if not batch_log_probs:
            continue

        rewards = np.array(batch_rewards)
        baselines = np.array(batch_baselines)
        advantages = rewards - baselines

        policy_loss = torch.tensor(0.0, device=device)
        for lps, adv in zip(batch_log_probs, advantages):
            policy_loss = policy_loss - lps.sum() * adv
        policy_loss = policy_loss / len(batch_log_probs)

        optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_reward += np.mean(batch_rewards)
        n += 1

    return -total_reward / max(n, 1)  # return avg cost


@torch.no_grad()
def evaluate_on_instances(model, instances, device, k, repair_steps=20):
    """Evaluate model on a set of instances."""
    model.eval()
    pre_costs, post_costs, n_frags = [], [], []

    for inst in instances:
        state = rollout_greedy(model, inst, device, k=k,
                               max_steps=inst['n_customers'] * 2)
        pre_costs.append(state.total_cost())
        n_frags.append(state.n_fragments)
        state = local_search_repair(state, max_steps=repair_steps)
        post_costs.append(state.total_cost())

    return {
        'pre_repair': np.mean(pre_costs),
        'post_repair': np.mean(post_costs),
        'n_fragments': np.mean(n_frags),
    }


def savings_baseline(instances, repair_steps=20):
    """Compute savings baseline costs."""
    costs = []
    for inst in instances:
        s = FragmentState.from_singletons(inst)
        s = project_savings(s, max_steps=inst['n_customers'] * 2)
        s = local_search_repair(s, max_steps=repair_steps)
        costs.append(s.total_cost())
    return np.mean(costs)


def run_experiment(model_name, model, train_instances, val_instances,
                   shifted_instances_dict, device, args):
    """Train one model and evaluate on standard + shifted instances."""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  {model_name}: {n_params:,} params")

    best_val = float('inf')
    results = {'epochs': [], 'val': [], 'shifts': {}}

    for epoch in range(1, args.n_epochs + 1):
        avg_cost = train_one_epoch(
            model, optimizer, train_instances, device, k=args.k,
            steps=args.steps_per_epoch, batch_size=args.batch_size,
            temperature=args.temperature)

        if epoch % args.eval_freq == 0 or epoch == args.n_epochs:
            # Standard validation
            val_diag = evaluate_on_instances(
                model, val_instances, device, k=args.k)
            results['epochs'].append(epoch)
            results['val'].append(val_diag)

            print(f"    Ep {epoch:3d}: train_cost={avg_cost:.3f} "
                  f"val_pre={val_diag['pre_repair']:.3f} "
                  f"val_post={val_diag['post_repair']:.3f} "
                  f"frags={val_diag['n_fragments']:.1f}")

            if val_diag['post_repair'] < best_val:
                best_val = val_diag['post_repair']
                best_state = {k: v.cpu().clone() for k, v in
                              model.state_dict().items()}

    # Restore best checkpoint
    model.load_state_dict(best_state)

    # Evaluate on constraint shifts
    print(f"    Evaluating constraint shifts...")
    for shift_name, shift_insts in shifted_instances_dict.items():
        shift_diag = evaluate_on_instances(
            model, shift_insts, device, k=args.k)
        results['shifts'][shift_name] = shift_diag
        print(f"    {shift_name}: pre={shift_diag['pre_repair']:.3f} "
              f"post={shift_diag['post_repair']:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_customers', type=int, default=50)
    parser.add_argument('--n_train', type=int, default=200)
    parser.add_argument('--n_val', type=int, default=20)
    parser.add_argument('--n_epochs', type=int, default=30)
    parser.add_argument('--steps_per_epoch', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--hidden_dim', type=int, default=96)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    device = torch.device(args.device)
    N = args.n_customers
    print(f"=== Constraint-Native Ablation: CVRP-{N} ===")
    print(f"Device: {device}")

    # Generate data
    print(f"Generating instances...")
    train_instances = generate_cvrp_dataset(N, args.n_train, seed=42, capacity_ratio=0.8)
    val_instances = generate_cvrp_dataset(N, args.n_val, seed=99999, capacity_ratio=0.8)

    # Constraint-shifted test sets (same coords/demands, different capacity)
    shifted = {}
    for ratio in [0.5, 0.6, 1.0, 1.2]:
        shifted[f'cap={ratio}'] = generate_cvrp_dataset(
            N, args.n_val, seed=99999, capacity_ratio=ratio)

    # Savings baselines
    print("Computing savings baselines...")
    sav_standard = savings_baseline(val_instances)
    print(f"  Standard (cap=0.8): {sav_standard:.4f}")
    for name, insts in shifted.items():
        sav = savings_baseline(insts)
        print(f"  {name}: {sav:.4f}")

    # ── Model A: Plain GNN ──
    model_a = FragmentGNN(
        n_node_feat=12, n_edge_feat=3,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers,
    ).to(device)

    results_a = run_experiment(
        "Plain GNN", model_a, train_instances, val_instances,
        shifted, device, args)

    # ── Model B: Constraint-Native GNN ──
    model_b = ConstraintNativeGNN(
        n_node_feat=12, n_edge_feat=3,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers,
        n_constraint_nodes=2,
    ).to(device)

    results_b = run_experiment(
        "Constraint-Native GNN", model_b, train_instances, val_instances,
        shifted, device, args)

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"  ABLATION SUMMARY: CVRP-{N}")
    print(f"{'='*70}")

    # Parameter counts
    n_a = sum(p.numel() for p in model_a.parameters())
    n_b = sum(p.numel() for p in model_b.parameters())
    print(f"  Params: Plain={n_a:,}  Constraint-Native={n_b:,}  ratio={n_b/n_a:.2f}x")

    # Standard validation
    best_a = min(results_a['val'], key=lambda d: d['post_repair'])
    best_b = min(results_b['val'], key=lambda d: d['post_repair'])
    print(f"\n  Standard (cap=0.8):")
    print(f"    Savings:           {sav_standard:.4f}")
    print(f"    Plain GNN:         {best_a['post_repair']:.4f} "
          f"(pre={best_a['pre_repair']:.4f})")
    print(f"    Constraint-Native: {best_b['post_repair']:.4f} "
          f"(pre={best_b['pre_repair']:.4f})")

    # Constraint shifts
    print(f"\n  Constraint Shifts (post-repair):")
    print(f"  {'Regime':>10} | {'Savings':>10} | {'Plain':>10} | {'C-Native':>10} | {'Δ Plain':>10} | {'Δ C-Native':>10}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}")

    for name in sorted(shifted.keys()):
        sav = savings_baseline(shifted[name])
        a_post = results_a['shifts'][name]['post_repair']
        b_post = results_b['shifts'][name]['post_repair']
        gap_a = (a_post / sav - 1) * 100
        gap_b = (b_post / sav - 1) * 100

        print(f"  {name:>10} | {sav:10.4f} | {a_post:10.4f} | {b_post:10.4f} | "
              f"{gap_a:+9.2f}% | {gap_b:+9.2f}%")

    # Pre-repair comparison on shifts
    print(f"\n  Pre-Repair on Shifts:")
    print(f"  {'Regime':>10} | {'Plain pre':>10} | {'C-Native pre':>12} | {'Winner':>10}")
    print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*12}-+-{'-'*10}")

    for name in sorted(shifted.keys()):
        a_pre = results_a['shifts'][name]['pre_repair']
        b_pre = results_b['shifts'][name]['pre_repair']
        winner = "C-Native" if b_pre < a_pre else "Plain"
        print(f"  {name:>10} | {a_pre:10.4f} | {b_pre:12.4f} | {winner:>10}")


if __name__ == '__main__':
    main()
