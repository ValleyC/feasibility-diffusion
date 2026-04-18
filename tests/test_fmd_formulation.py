"""
Honest test: Does Boltzmann-target FMD training actually work?

Tests the core claim: a GNN trained on softmax(-delta/tau) targets can
predict which moves improve solutions WITHOUT computing deltas at inference.

Tests:
  1. Score-Delta Correlation: does the model's ranking match true delta ranking?
  2. Denoising Quality: model-guided search vs greedy local search vs random
  3. Stochastic Diversity: does sampling produce diverse solutions? Does best-of-K help?
  4. Time Conditioning: does the model behave differently at different t values?
  5. Boltzmann vs Hard Classification: is soft target better than our current approach?

Uses TSP-20 for speed. All comparisons use same seeds and starting points.

Usage:
    python -m tests.test_fmd_formulation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

from models.problem_configs import TSPConfig
from training.generic_trainer import GNNLayer, prepare_batch_item


# ─── Time-conditioned model ────────────────────────────────────

class FMDMoveScorer(nn.Module):
    """GNN move scorer with sinusoidal time conditioning."""

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


# ─── Data generation ───────────────────────────────────────────

def generate_training_data(config, manifold, instances, max_moves, n_quality_levels=5):
    """Generate (solution, moves, deltas, t) at various quality levels.

    Quality 0 = random tour, quality n-1 = well-optimized tour.
    t = 1 - quality (high t = bad solution = early in reverse process).
    """
    data = []
    for idx, inst in enumerate(instances):
        for ql in range(n_quality_levels):
            sol = manifold.sample_random(inst)
            n_improve = ql * 15
            for _ in range(n_improve):
                moves = manifold.enumerate_moves(sol, inst)
                if len(moves) == 0:
                    break
                deltas = np.array([manifold.move_delta(sol, m, inst) for m in moves])
                best = np.argmin(deltas)
                if deltas[best] >= -1e-10:
                    break
                sol = manifold.apply_move(sol, moves[best])

            moves = manifold.enumerate_moves(sol, inst)
            if len(moves) == 0 or len(moves) > max_moves:
                continue
            deltas = np.array([manifold.move_delta(sol, m, inst) for m in moves])
            t = 1.0 - ql / max(n_quality_levels - 1, 1)

            data.append({
                'idx': idx, 'sol': sol, 'moves': moves,
                'deltas': deltas, 't': t,
                'cost': manifold.cost(sol, inst),
            })
    return data


def boltzmann_target(deltas, tau):
    """softmax(-delta / tau): favor moves with negative delta (improving)."""
    logits = -deltas / max(tau, 1e-8)
    logits = logits - logits.max()
    probs = np.exp(logits)
    return probs / probs.sum()


# ─── Training ──────────────────────────────────────────────────

def train_model(model, config, instances, train_data, max_moves, device,
                mode='boltzmann', n_epochs=30, batch_size=16, lr=3e-4,
                tau_min=0.01, tau_max=1.0):
    """Train with Boltzmann soft targets or hard classification."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    n_data = len(train_data)

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_steps = min(100, n_data // batch_size)

        for step in range(n_steps):
            items = []
            max_edges = 0

            for _ in range(batch_size):
                d = train_data[np.random.randint(n_data)]
                t = d['t']
                tau_t = tau_min + t * (tau_max - tau_min)

                if mode == 'boltzmann':
                    target = boltzmann_target(d['deltas'], tau_t)
                else:
                    target = None
                    best_idx = int(np.argmin(d['deltas']))

                nf, ei, mn, mm, _ = prepare_batch_item(
                    config, d['sol'], instances[d['idx']], d['moves'], 0, t, max_moves
                )
                items.append((nf, ei, mn, mm, target, best_idx if mode != 'boltzmann' else 0, t))
                max_edges = max(max_edges, ei.shape[0])

            batch_nf, batch_ei, batch_mn, batch_mm = [], [], [], []
            batch_targets, batch_labels, batch_t = [], [], []

            for nf, ei, mn, mm, target, label, t in items:
                if ei.shape[0] < max_edges:
                    pad = np.zeros((max_edges - ei.shape[0], 2), dtype=np.int64)
                    ei = np.vstack([ei, pad])
                batch_nf.append(nf)
                batch_ei.append(ei)
                batch_mn.append(mn)
                batch_mm.append(mm)
                batch_t.append(t)

                if mode == 'boltzmann':
                    padded = np.zeros(max_moves)
                    padded[:len(target)] = target
                    batch_targets.append(padded)
                else:
                    batch_labels.append(min(label, max_moves - 1))

            nf_t = torch.tensor(np.stack(batch_nf), dtype=torch.float32, device=device)
            ei_t = torch.tensor(np.stack(batch_ei), dtype=torch.long, device=device)
            mn_t = torch.tensor(np.stack(batch_mn), dtype=torch.long, device=device)
            mm_t = torch.tensor(np.stack(batch_mm), dtype=torch.bool, device=device)
            t_t = torch.tensor(batch_t, dtype=torch.float32, device=device)

            scores = model(nf_t, ei_t, mn_t, mm_t, t_t)

            if mode == 'boltzmann':
                targets_t = torch.tensor(np.stack(batch_targets), dtype=torch.float32, device=device)
                log_probs = F.log_softmax(scores, dim=-1)
                # Mask out padded positions to avoid 0 * (-inf) = nan
                log_probs = log_probs.masked_fill(~mm_t, 0.0)
                loss = -(targets_t * log_probs).sum(dim=-1).mean()
            else:
                labels_t = torch.tensor(batch_labels, dtype=torch.long, device=device)
                loss = F.cross_entropy(scores, labels_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}: loss={epoch_loss / n_steps:.4f}")


# ─── Test functions ────────────────────────────────────────────

def model_inference(model, config, manifold, inst, moves, sol, t_val, max_moves, device):
    """Single forward pass, return scores as numpy array."""
    nf, ei, mn, mm, _ = prepare_batch_item(config, sol, inst, moves, 0, t_val, max_moves)
    nf_t = torch.tensor(nf, dtype=torch.float32, device=device).unsqueeze(0)
    ei_t = torch.tensor(ei, dtype=torch.long, device=device).unsqueeze(0)
    mn_t = torch.tensor(mn, dtype=torch.long, device=device).unsqueeze(0)
    mm_t = torch.tensor(mm, dtype=torch.bool, device=device).unsqueeze(0)
    t_t = torch.tensor([t_val], dtype=torch.float32, device=device)
    with torch.no_grad():
        scores = model(nf_t, ei_t, mn_t, mm_t, t_t)
    return scores[0, :len(moves)].cpu().numpy()


def test_1_correlation(model, config, manifold, val_instances, max_moves, device):
    """Does the model's score ranking match the true delta ranking?"""
    model.eval()
    correlations = []
    top1_improving = 0
    total = 0

    for inst in val_instances:
        sol = manifold.sample_random(inst)
        for _ in range(np.random.randint(0, 20)):
            moves = manifold.enumerate_moves(sol, inst)
            if not moves:
                break
            deltas = np.array([manifold.move_delta(sol, m, inst) for m in moves])
            best = np.argmin(deltas)
            if deltas[best] >= -1e-10:
                break
            sol = manifold.apply_move(sol, moves[best])

        moves = manifold.enumerate_moves(sol, inst)
        if len(moves) < 3 or len(moves) > max_moves:
            continue
        deltas = np.array([manifold.move_delta(sol, m, inst) for m in moves])

        scores = model_inference(model, config, manifold, inst, moves, sol, 0.3, max_moves, device)

        corr, _ = spearmanr(scores, -deltas)
        if not np.isnan(corr):
            correlations.append(corr)

        top1_idx = np.argmax(scores)
        if deltas[top1_idx] < -1e-10:
            top1_improving += 1
        total += 1

    return {
        'spearman_corr': np.mean(correlations) if correlations else 0,
        'top1_improving_rate': top1_improving / max(total, 1),
        'n_tested': total,
    }


def run_model_denoising(model, config, manifold, inst, max_moves, device,
                        n_steps, stochastic=False, temperature=0.3, seed=None):
    """Run denoising from random start, return final cost."""
    if seed is not None:
        np.random.seed(seed)
    sol = manifold.sample_random(inst)

    for step in range(n_steps):
        moves = manifold.enumerate_moves(sol, inst)
        if len(moves) == 0 or len(moves) > max_moves:
            break
        t_val = 1.0 - step / max(n_steps - 1, 1)
        scores = model_inference(model, config, manifold, inst, moves, sol, t_val, max_moves, device)

        if stochastic:
            temp = max(temperature * t_val, 0.05)
            probs = np.exp(scores / temp - np.max(scores / temp))
            probs = probs / probs.sum()
            best = np.random.choice(len(moves), p=probs)
        else:
            best = int(np.argmax(scores))

        if best < len(moves):
            d = manifold.move_delta(sol, moves[best], inst)
            if d < -1e-10:
                sol = manifold.apply_move(sol, moves[best])

    return manifold.cost(sol, inst), sol


def run_greedy_ls(manifold, inst, n_steps, seed=None):
    """Greedy local search: always pick best delta. Oracle baseline."""
    if seed is not None:
        np.random.seed(seed)
    sol = manifold.sample_random(inst)

    for _ in range(n_steps):
        moves = manifold.enumerate_moves(sol, inst)
        if not moves:
            break
        deltas = np.array([manifold.move_delta(sol, m, inst) for m in moves])
        best = np.argmin(deltas)
        if deltas[best] >= -1e-10:
            break
        sol = manifold.apply_move(sol, moves[best])

    return manifold.cost(sol, inst), sol


def run_random_improving(manifold, inst, n_steps, seed=None):
    """Random improving walk: pick random move, apply if improving."""
    if seed is not None:
        np.random.seed(seed)
    sol = manifold.sample_random(inst)

    for _ in range(n_steps):
        moves = manifold.enumerate_moves(sol, inst)
        if not moves:
            break
        m = moves[np.random.randint(len(moves))]
        d = manifold.move_delta(sol, m, inst)
        if d < -1e-10:
            sol = manifold.apply_move(sol, m)

    return manifold.cost(sol, inst), sol


def test_2_denoising_quality(model, config, manifold, val_instances, max_moves, device,
                             n_steps=60):
    """Model-guided search vs greedy vs random. Same starting seeds."""
    model.eval()
    model_costs, greedy_costs, random_costs = [], [], []

    for i, inst in enumerate(val_instances):
        seed = 77777 + i
        mc, _ = run_model_denoising(model, config, manifold, inst, max_moves, device,
                                    n_steps, stochastic=False, seed=seed)
        gc, _ = run_greedy_ls(manifold, inst, n_steps, seed=seed)
        rc, _ = run_random_improving(manifold, inst, n_steps, seed=seed)
        model_costs.append(mc)
        greedy_costs.append(gc)
        random_costs.append(rc)

    return {
        'model_avg': np.mean(model_costs),
        'greedy_avg': np.mean(greedy_costs),
        'random_avg': np.mean(random_costs),
        'model_vs_greedy_pct': (np.mean(model_costs) / np.mean(greedy_costs) - 1) * 100,
        'random_vs_greedy_pct': (np.mean(random_costs) / np.mean(greedy_costs) - 1) * 100,
    }


def test_3_diversity(model, config, manifold, val_instances, max_moves, device,
                     K=16, n_steps=60):
    """Stochastic sampling diversity and best-of-K improvement."""
    model.eval()
    greedy_costs, best_of_k, single_sample = [], [], []
    unique_counts = []

    for i, inst in enumerate(val_instances[:10]):
        gc, _ = run_greedy_ls(manifold, inst, n_steps * 2, seed=77777 + i)
        greedy_costs.append(gc)

        sample_costs = []
        for k in range(K):
            mc, _ = run_model_denoising(model, config, manifold, inst, max_moves, device,
                                        n_steps, stochastic=True, temperature=0.5,
                                        seed=88888 + i * K + k)
            sample_costs.append(mc)

        best_of_k.append(min(sample_costs))
        single_sample.append(sample_costs[0])
        unique_counts.append(len(set(round(c, 4) for c in sample_costs)))

    return {
        'greedy_avg': np.mean(greedy_costs),
        'single_sample_avg': np.mean(single_sample),
        'best_of_K_avg': np.mean(best_of_k),
        'avg_unique_out_of_K': np.mean(unique_counts),
        'best_of_K_vs_greedy_pct': (np.mean(best_of_k) / np.mean(greedy_costs) - 1) * 100,
        'single_vs_greedy_pct': (np.mean(single_sample) / np.mean(greedy_costs) - 1) * 100,
    }


def test_4_time_conditioning(model, config, manifold, val_instances, max_moves, device):
    """Does model output entropy increase with t? (exploratory at high t, focused at low t)"""
    model.eval()
    entropies = {0.1: [], 0.5: [], 0.9: []}

    for inst in val_instances[:20]:
        sol = manifold.sample_random(inst)
        for _ in range(10):
            moves = manifold.enumerate_moves(sol, inst)
            if not moves:
                break
            deltas = np.array([manifold.move_delta(sol, m, inst) for m in moves])
            best = np.argmin(deltas)
            if deltas[best] >= -1e-10:
                break
            sol = manifold.apply_move(sol, moves[best])

        moves = manifold.enumerate_moves(sol, inst)
        if len(moves) < 3 or len(moves) > max_moves:
            continue

        for t_val in [0.1, 0.5, 0.9]:
            scores = model_inference(model, config, manifold, inst, moves, sol, t_val, max_moves, device)
            probs = np.exp(scores - scores.max())
            probs = probs / probs.sum()
            entropy = -np.sum(probs * np.log(probs + 1e-10))
            entropies[t_val].append(entropy)

    return {
        't=0.1_entropy': np.mean(entropies[0.1]),
        't=0.5_entropy': np.mean(entropies[0.5]),
        't=0.9_entropy': np.mean(entropies[0.9]),
        'monotonic_increasing': (np.mean(entropies[0.1]) < np.mean(entropies[0.5])
                                 < np.mean(entropies[0.9])),
    }


def test_5_boltzmann_vs_hard(config, manifold, train_instances, val_instances, max_moves, device,
                             train_data, tau_min=0.1, tau_max=1.0, n_steps=60):
    """Train two models (Boltzmann vs hard classification), compare performance."""
    results = {}

    for mode in ['boltzmann', 'hard']:
        print(f"\n  Training {mode} model...")
        model = FMDMoveScorer(
            n_node_features=config.n_node_features,
            hidden_dim=48, n_layers=3,
        ).to(device)
        train_model(model, config, train_instances, train_data, max_moves, device,
                    mode=mode, n_epochs=40, batch_size=16, lr=3e-4,
                    tau_min=tau_min, tau_max=tau_max)

        model.eval()
        costs = []
        for i, inst in enumerate(val_instances):
            mc, _ = run_model_denoising(model, config, manifold, inst, max_moves, device,
                                        n_steps, stochastic=False, seed=77777 + i)
            costs.append(mc)
        results[mode] = np.mean(costs)

    greedy_costs = []
    for i, inst in enumerate(val_instances):
        gc, _ = run_greedy_ls(manifold, inst, n_steps, seed=77777 + i)
        greedy_costs.append(gc)
    greedy_avg = np.mean(greedy_costs)

    return {
        'boltzmann_avg': results['boltzmann'],
        'hard_avg': results['hard'],
        'greedy_avg': greedy_avg,
        'boltzmann_vs_greedy_pct': (results['boltzmann'] / greedy_avg - 1) * 100,
        'hard_vs_greedy_pct': (results['hard'] / greedy_avg - 1) * 100,
    }


# ─── Main ──────────────────────────────────────────────────────

def main():
    device = torch.device('cpu')
    N = 20
    max_moves = 500
    n_train = 100
    n_val = 30

    print("=" * 65)
    print("FMD Formulation Test — TSP-20")
    print("=" * 65)

    config = TSPConfig()
    manifold = config.create_manifold()

    print(f"\nSetup: N={N}, {n_train} train instances, {n_val} val instances")
    train_instances = [config.create_instance(N, seed=42 + i) for i in range(n_train)]
    val_instances = [config.create_instance(N, seed=99999 + i) for i in range(n_val)]

    # Calibrate tau from typical delta magnitudes
    sample_deltas = []
    for inst in train_instances[:10]:
        sol = manifold.sample_random(inst)
        moves = manifold.enumerate_moves(sol, inst)
        for m in moves[:50]:
            sample_deltas.append(abs(manifold.move_delta(sol, m, inst)))
    delta_scale = np.median(sample_deltas) if sample_deltas else 0.1
    tau_min = delta_scale * 0.5   # peaked but not one-hot at t=0
    tau_max = delta_scale * 5.0   # soft/exploratory at t=1
    print(f"Delta scale: median |delta| = {delta_scale:.4f}")
    print(f"Temperature range: tau_min={tau_min:.4f}, tau_max={tau_max:.4f}")

    # Generate training data
    print("\nGenerating training data (5 quality levels per instance)...")
    t0 = time.time()
    train_data = generate_training_data(config, manifold, train_instances, max_moves)
    print(f"  {len(train_data)} training samples in {time.time() - t0:.1f}s")

    # Train main model
    print("\nTraining FMD model (Boltzmann targets)...")
    model = FMDMoveScorer(
        n_node_features=config.n_node_features,
        hidden_dim=48, n_layers=3,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {n_params:,} params")
    train_model(model, config, train_instances, train_data, max_moves, device,
                mode='boltzmann', n_epochs=40, batch_size=16, lr=3e-4,
                tau_min=tau_min, tau_max=tau_max)

    # ─── Test 1 ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("TEST 1: Score-Delta Correlation")
    print("  Can the model rank moves correctly without computing deltas?")
    print("-" * 65)
    r1 = test_1_correlation(model, config, manifold, val_instances, max_moves, device)
    print(f"  Spearman correlation (score vs -delta): {r1['spearman_corr']:.3f}")
    print(f"  Top-1 scored move is improving:         {r1['top1_improving_rate']:.1%}")
    print(f"  Tested on {r1['n_tested']} solution states")

    pass1 = r1['spearman_corr'] > 0.3 and r1['top1_improving_rate'] > 0.5
    print(f"  {'PASS' if pass1 else 'FAIL'}: correlation > 0.3 and top-1 improving > 50%")

    # ─── Test 2 ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("TEST 2: Denoising Quality")
    print("  Model-guided search vs greedy local search vs random walk")
    print("-" * 65)
    r2 = test_2_denoising_quality(model, config, manifold, val_instances, max_moves, device)
    print(f"  Greedy local search (oracle): {r2['greedy_avg']:.4f}")
    print(f"  FMD model (no delta comp):    {r2['model_avg']:.4f}  ({r2['model_vs_greedy_pct']:+.1f}%)")
    print(f"  Random improving walk:        {r2['random_avg']:.4f}  ({r2['random_vs_greedy_pct']:+.1f}%)")

    pass2 = r2['model_vs_greedy_pct'] < r2['random_vs_greedy_pct']
    print(f"  {'PASS' if pass2 else 'FAIL'}: model beats random walk")

    pass2b = r2['model_vs_greedy_pct'] < 20
    print(f"  {'PASS' if pass2b else 'FAIL'}: model within 20% of greedy")

    # ─── Test 3 ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("TEST 3: Stochastic Sampling Diversity")
    print("  K=16 stochastic samples: diversity and best-of-K improvement")
    print("-" * 65)
    r3 = test_3_diversity(model, config, manifold, val_instances, max_moves, device)
    print(f"  Greedy local search:     {r3['greedy_avg']:.4f}")
    print(f"  Single stochastic:       {r3['single_sample_avg']:.4f}  ({r3['single_vs_greedy_pct']:+.1f}%)")
    print(f"  Best-of-16:              {r3['best_of_K_avg']:.4f}  ({r3['best_of_K_vs_greedy_pct']:+.1f}%)")
    print(f"  Unique solutions (of 16): {r3['avg_unique_out_of_K']:.1f}")

    pass3 = r3['avg_unique_out_of_K'] > 3
    print(f"  {'PASS' if pass3 else 'FAIL'}: diversity (> 3 unique solutions)")

    pass3b = r3['best_of_K_vs_greedy_pct'] < r3['single_vs_greedy_pct']
    print(f"  {'PASS' if pass3b else 'FAIL'}: best-of-K improves over single sample")

    # ─── Test 4 ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("TEST 4: Time Conditioning")
    print("  Does model entropy increase with t? (exploratory early, focused late)")
    print("-" * 65)
    r4 = test_4_time_conditioning(model, config, manifold, val_instances, max_moves, device)
    print(f"  Entropy at t=0.1 (clean):   {r4['t=0.1_entropy']:.3f}")
    print(f"  Entropy at t=0.5 (medium):  {r4['t=0.5_entropy']:.3f}")
    print(f"  Entropy at t=0.9 (noisy):   {r4['t=0.9_entropy']:.3f}")
    print(f"  Monotonically increasing:   {r4['monotonic_increasing']}")

    pass4 = r4['t=0.1_entropy'] < r4['t=0.9_entropy']
    print(f"  {'PASS' if pass4 else 'FAIL'}: entropy higher at high t than low t")

    # ─── Test 5 ─────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("TEST 5: Boltzmann vs Hard Classification")
    print("  Compare soft (FMD) vs hard (current approach) training targets")
    print("-" * 65)
    r5 = test_5_boltzmann_vs_hard(config, manifold, train_instances, val_instances,
                                  max_moves, device, train_data,
                                  tau_min=tau_min, tau_max=tau_max)
    print(f"\n  Greedy local search:  {r5['greedy_avg']:.4f}")
    print(f"  Boltzmann (FMD):      {r5['boltzmann_avg']:.4f}  ({r5['boltzmann_vs_greedy_pct']:+.1f}%)")
    print(f"  Hard classification:  {r5['hard_avg']:.4f}  ({r5['hard_vs_greedy_pct']:+.1f}%)")

    pass5 = abs(r5['boltzmann_vs_greedy_pct']) <= abs(r5['hard_vs_greedy_pct']) + 5
    print(f"  {'PASS' if pass5 else 'FAIL'}: Boltzmann competitive with hard (within 5%)")

    # ─── Summary ────────────────────────────────────────────
    print("\n" + "=" * 65)
    all_pass = [pass1, pass2, pass2b, pass3, pass3b, pass4, pass5]
    n_pass = sum(all_pass)
    print(f"RESULTS: {n_pass}/{len(all_pass)} tests passed")

    if n_pass == len(all_pass):
        print("ALL TESTS PASSED — FMD formulation is viable")
    else:
        print("Some tests failed — see details above")
        failed = []
        if not pass1: failed.append("1: score-delta correlation too low")
        if not pass2: failed.append("2a: model worse than random walk")
        if not pass2b: failed.append("2b: model too far from greedy")
        if not pass3: failed.append("3a: insufficient diversity")
        if not pass3b: failed.append("3b: best-of-K doesn't help")
        if not pass4: failed.append("4: time conditioning not working")
        if not pass5: failed.append("5: Boltzmann much worse than hard")
        for f in failed:
            print(f"  FAILED: {f}")

    print("=" * 65)


if __name__ == '__main__':
    main()
