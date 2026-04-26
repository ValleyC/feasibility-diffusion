"""
Label-Free REINFORCE Trainer for Fragment Merge Scoring.

No elite buffer, no supervised adjacency imitation. The model learns
purely from decoded route cost via policy gradient.

Each training step:
  1. Sample an instance
  2. Run the projector as a stochastic policy (sample from softmax of scores)
  3. Decode full routes
  4. Reward = -cost (lower cost = higher reward)
  5. REINFORCE gradient on merge log-probs

Baseline: greedy rollout with the current model (no sampling).

Usage:
    python -m training.rl_trainer --n_customers 1000 --device cuda:0
"""

import sys, os, argparse, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from data.cvrp_generator import generate_cvrp_dataset
from route_objects.fragment import check_merge_feasible, simulate_route
from route_objects.fragment_state import FragmentState, build_fragment_graph
from route_objects.projector import (
    project, project_savings, local_search_repair, finalize_routes,
)
from models.fragment_gnn import FragmentGNN


def rollout_stochastic(model, instance, device, k=20, temperature=1.0,
                       max_steps=2000):
    """Run stochastic projector collecting log-probs for REINFORCE.

    Returns:
        state: final FragmentState
        log_probs: list of log-prob tensors for each accepted merge
    """
    state = FragmentState.from_singletons(instance)
    log_probs = []

    for step in range(max_steps):
        if state.n_fragments <= 1:
            break

        node_feat, edge_index, edge_feat = build_fragment_graph(state, k=k)
        if len(edge_index) == 0:
            break

        nf_t = torch.tensor(node_feat, dtype=torch.float32, device=device)
        ei_t = torch.tensor(edge_index, dtype=torch.long, device=device)
        ef_t = torch.tensor(edge_feat, dtype=torch.float32, device=device)

        merge_scores, risk_scores = model(nf_t, ei_t, ef_t)

        # Mask high-risk edges
        mask = risk_scores < 0.8
        if not mask.any():
            break

        masked_scores = merge_scores.clone()
        masked_scores[~mask] = float('-inf')

        # Sample from softmax
        probs = F.softmax(masked_scores / temperature, dim=0)
        if torch.isnan(probs).any() or probs.sum() < 1e-8:
            break

        action = torch.multinomial(probs, 1).item()
        log_probs.append(torch.log(probs[action] + 1e-10))

        src_idx = edge_index[action, 0]
        dst_idx = edge_index[action, 1]

        feasible, _ = check_merge_feasible(
            state.fragments[src_idx], state.fragments[dst_idx],
            instance, orientation=0)

        if feasible:
            state = state.apply_merge(src_idx, dst_idx, orientation=0)
        else:
            # Infeasible sample — still record the log-prob (it will get
            # negative reward signal), but don't apply the merge
            break

    return state, log_probs


@torch.no_grad()
def rollout_greedy(model, instance, device, k=20, max_steps=2000):
    """Greedy rollout for baseline cost."""
    return project(model, FragmentState.from_singletons(instance),
                   device, k=k, max_steps=max_steps)


@torch.no_grad()
def evaluate(model, instances, device, k=20, n_eval=20, repair_steps=30):
    """Evaluate on validation instances."""
    model.eval()
    costs_model, costs_savings, pre_costs = [], [], []
    n_frags = []

    for vi in range(min(n_eval, len(instances))):
        inst = instances[vi]

        # Model greedy
        state = rollout_greedy(model, inst, device, k=k)
        pre_costs.append(state.total_cost())
        n_frags.append(state.n_fragments)
        state = local_search_repair(state, max_steps=repair_steps)
        costs_model.append(state.total_cost())

        # Savings baseline
        state_s = FragmentState.from_singletons(inst)
        state_s = project_savings(state_s, max_steps=2000)
        state_s = local_search_repair(state_s, max_steps=repair_steps)
        costs_savings.append(state_s.total_cost())

    return {
        'model_cost': np.mean(costs_model),
        'savings_cost': np.mean(costs_savings),
        'pre_repair': np.mean(pre_costs),
        'n_fragments': np.mean(n_frags),
        'gap': (np.mean(costs_model) / np.mean(costs_savings) - 1) * 100,
    }


def train(args):
    device = torch.device(args.device)
    os.makedirs(args.ckpt_dir, exist_ok=True)

    print(f"Generating {args.n_instances} CVRP-{args.n_customers} instances...")
    train_instances = generate_cvrp_dataset(
        args.n_customers, args.n_instances, seed=42,
        capacity_ratio=args.capacity_ratio)
    val_instances = generate_cvrp_dataset(
        args.n_customers, args.n_val, seed=99999,
        capacity_ratio=args.capacity_ratio)

    # Compute savings baseline on validation
    print("Computing savings baseline...")
    val_savings = []
    for inst in val_instances[:args.n_eval]:
        s = FragmentState.from_singletons(inst)
        s = project_savings(s, max_steps=args.n_customers * 2)
        s = local_search_repair(s, max_steps=args.repair_steps)
        val_savings.append(s.total_cost())
    ref_cost = np.mean(val_savings)
    print(f"  Val savings: {ref_cost:.4f}")

    # Model
    model = FragmentGNN(
        n_node_feat=12, n_edge_feat=3,
        hidden_dim=args.hidden_dim, n_layers=args.n_layers,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  FragmentGNN: {n_params:,} params")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.n_epochs)

    best_gap = float('inf')

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        epoch_reward = 0.0
        epoch_loss = 0.0
        epoch_baseline = 0.0
        n_steps = 0

        pbar = tqdm(range(args.steps_per_epoch), desc=f"Epoch {epoch}")
        for step in pbar:
            # Sample a batch of instances
            batch_rewards = []
            batch_baselines = []
            batch_log_probs = []

            for _ in range(args.batch_size):
                idx = np.random.randint(len(train_instances))
                inst = train_instances[idx]

                # Stochastic rollout (for gradient)
                state_s, lps = rollout_stochastic(
                    model, inst, device, k=args.k,
                    temperature=args.temperature)
                if not lps:
                    continue
                cost_s = state_s.total_cost()

                # Greedy rollout (baseline)
                state_g = rollout_greedy(model, inst, device, k=args.k)
                cost_g = state_g.total_cost()

                batch_rewards.append(-cost_s)
                batch_baselines.append(-cost_g)
                batch_log_probs.append(torch.stack(lps))

            if not batch_log_probs:
                continue

            # REINFORCE with greedy baseline
            rewards = np.array(batch_rewards)
            baselines = np.array(batch_baselines)
            advantages = rewards - baselines  # positive = stochastic was better

            policy_loss = torch.tensor(0.0, device=device)
            for lps, adv in zip(batch_log_probs, advantages):
                policy_loss = policy_loss - lps.sum() * adv
            policy_loss = policy_loss / len(batch_log_probs)

            # Entropy bonus for exploration
            entropy_bonus = torch.tensor(0.0, device=device)
            for lps in batch_log_probs:
                entropy_bonus = entropy_bonus - lps.mean()
            entropy_bonus = entropy_bonus / len(batch_log_probs) * args.entropy_coeff

            loss = policy_loss - entropy_bonus

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_reward += np.mean(batch_rewards)
            epoch_baseline += np.mean(batch_baselines)
            epoch_loss += loss.item()
            n_steps += 1

            avg_cost = -np.mean(batch_rewards)
            pbar.set_postfix(cost=f"{avg_cost:.2f}",
                             loss=f"{loss.item():.3f}")

        scheduler.step()
        if n_steps > 0:
            avg_r = -epoch_reward / n_steps
            avg_b = -epoch_baseline / n_steps
            print(f"  Epoch {epoch}: cost={avg_r:.4f} baseline={avg_b:.4f} "
                  f"loss={epoch_loss/n_steps:.4f}")

        # Evaluate
        if epoch % args.eval_freq == 0 or epoch == args.n_epochs:
            diag = evaluate(model, val_instances, device, k=args.k,
                            n_eval=args.n_eval, repair_steps=args.repair_steps)

            print(f"  Val model:   {diag['model_cost']:.4f} ({diag['gap']:+.2f}% vs savings)")
            print(f"  Val savings: {diag['savings_cost']:.4f}")
            print(f"  Pre-repair:  {diag['pre_repair']:.4f}")
            print(f"  Avg frags:   {diag['n_fragments']:.1f}")

            if diag['gap'] < best_gap:
                best_gap = diag['gap']
                path = os.path.join(args.ckpt_dir, 'best.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'gap': diag['gap'],
                    'args': vars(args),
                }, path)
                print(f"  [best] gap={diag['gap']:+.2f}% -> {path}")

    print(f"\nDone. Best gap vs savings: {best_gap:+.2f}%")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Label-Free RL Fragment Trainer')
    parser.add_argument('--n_customers', type=int, default=1000)
    parser.add_argument('--n_instances', type=int, default=100)
    parser.add_argument('--n_val', type=int, default=20)
    parser.add_argument('--n_eval', type=int, default=10)
    parser.add_argument('--capacity_ratio', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--steps_per_epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--temperature', type=float, default=2.0)
    parser.add_argument('--entropy_coeff', type=float, default=0.01)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=6)
    parser.add_argument('--k', type=int, default=50)
    parser.add_argument('--repair_steps', type=int, default=20)
    parser.add_argument('--eval_freq', type=int, default=5)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/rl_cvrp')
    args = parser.parse_args()
    train(args)
