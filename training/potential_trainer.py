"""
State-Potential Diffusion on the Feasibility Manifold.

Learns a time-dependent scalar potential f_θ(x, t) ≈ log p_t(x) over
feasible states. Reverse transition logits are DERIVED from potential
differences:

    logit(x → y via move m) = Δf(m) ≈ f(y, t−1) − f(x, t)

The model has two heads sharing a GNN backbone:
  1. State potential head: GNN → mean pool → MLP → scalar f(x, t)
  2. Move difference head: 4 boundary nodes → scalar Δf(m)

Training uses three losses:
  1. Contrastive potential: f(x_t, t) > f(x_neg, t)
     (trajectory samples are more likely than random tours at time t)
  2. Trajectory ranking: f(x_{t−1}, t−1) > f(x_t, t−1)
     (predecessor is more likely at time t−1 than the corrupted state)
  3. Move consistency: Δf(inverse_move) should be largest among all moves,
     AND for K sampled moves, Δf(m_k) ≈ f(y_k, t−1) − f(x_t, t) [stop-grad]
     (move scorer is anchored by the state potential)

Inference: sample from softmax(Δf / temperature), return terminal state.

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

class StatePotentialModel(nn.Module):
    """
    Time-dependent state potential f_θ(x, t) ≈ log p_t(x).

    Two heads sharing a GNN backbone:
      1. f(x, t): scalar state potential (global mean pool → MLP → R)
      2. Δf(m): potential difference per move (4 boundary nodes → MLP → R)

    The reverse transition logit for move m is Δf(m) ≈ f(y_m, t−1) − f(x, t).
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

        # State potential head: mean pool → scalar
        self.potential_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Move difference head: 4 boundary node embeddings → scalar Δf
        self.diff_head = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def time_encoding(self, t):
        freqs = torch.exp(torch.arange(0, 8, device=t.device).float()
                          * (-np.log(10000) / 8))
        args = t.unsqueeze(-1) * freqs
        return torch.cat([args.sin(), args.cos()], dim=-1)

    def encode(self, node_features, edge_index, t):
        """Run GNN, return node embeddings h and state potential f."""
        B, N, _ = node_features.shape
        t_emb = self.time_embed(self.time_encoding(t))
        h = self.embed(node_features) + t_emb.unsqueeze(1)
        for layer in self.gnn:
            h = layer(h, edge_index)

        # State potential: mean pool → scalar
        f = self.potential_head(h.mean(dim=1)).squeeze(-1)  # (B,)
        return h, f

    def move_diffs(self, h, move_nodes, move_mask):
        """Compute Δf for each move from 4 boundary node embeddings."""
        B, N, D = h.shape
        M = move_nodes.shape[1]
        idx = move_nodes.clamp(0, N - 1)
        h_parts = []
        for p in range(4):
            node_idx = idx[:, :, p].unsqueeze(-1).expand(-1, -1, D)
            h_parts.append(h.gather(1, node_idx))
        h_moves = torch.cat(h_parts, dim=-1)  # (B, M, 4D)
        diffs = self.diff_head(h_moves).squeeze(-1)  # (B, M)
        return diffs.masked_fill(~move_mask, float('-inf'))

    def forward(self, node_features, edge_index, move_nodes, move_mask, t):
        """Full forward: state potential + move diffs."""
        h, f = self.encode(node_features, edge_index, t)
        diffs = self.move_diffs(h, move_nodes, move_mask)
        return f, diffs

    def forward_potential_only(self, node_features, edge_index, t):
        """Just the state potential (no moves). Used for contrastive training."""
        _, f = self.encode(node_features, edge_index, t)
        return f


# ─── Training data generation ──────────────────────────────────

def generate_potential_data(config, manifold, instances, clean_solutions,
                            t_max, max_moves, n_samples_per_instance=10):
    """Generate training data: (x_t, t, x_prev, inverse_move_idx, moves).

    For each (x_0, instance), run forward trajectories and store:
      - x_t: noisy solution at step t
      - x_prev: predecessor x_{t-1} (for trajectory ranking)
      - inv_idx: index of inverse move in enumerate_moves(x_t)
      - moves: available moves at x_t
    """
    pool = []
    for idx in tqdm(range(len(instances)), desc="  Generating data"):
        inst = instances[idx]
        x_0 = clean_solutions[idx]

        for _ in range(n_samples_per_instance):
            t = np.random.randint(1, t_max + 1)

            # Forward: x_0 → x_1 → ... → x_t, recording moves
            x_t, applied_moves = scramble_with_moves(manifold, x_0, inst, t)

            if not applied_moves:
                continue

            last_move = applied_moves[-1]

            # x_{t-1} = apply inverse of last move. For 2-opt, inverse = same move.
            x_prev = manifold.apply_move(x_t, last_move)

            # Enumerate moves at x_t
            moves = manifold.enumerate_moves(x_t, inst)
            if not moves:
                continue

            # Find inverse move index
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
                insert_pos = 0
                for k, ki in enumerate(keep):
                    if ki > inv_idx:
                        insert_pos = k
                        break
                    insert_pos = k + 1
                keep.insert(insert_pos, inv_idx)
                moves = [moves[i] for i in keep]
                inv_idx = insert_pos

            pool.append({
                'idx': idx,
                'x_t': x_t.copy() if hasattr(x_t, 'copy') else x_t,
                'x_prev': x_prev.copy() if hasattr(x_prev, 'copy') else x_prev,
                't': t,
                'moves': moves,
                'inv_idx': inv_idx,
            })

    print(f"  {len(pool)} samples ({len(pool)/len(instances):.1f} per instance)")
    return pool


# ─── Feature building helpers ─────────────────────────────────

def build_state_tensors(config, solution, instance, t_norm, device):
    """Build node features + edge index for a single state (no moves)."""
    nf = config.build_node_features(solution, instance, t_norm)
    ei = config.build_edges(solution, instance)
    nf_t = torch.tensor(nf, dtype=torch.float32, device=device).unsqueeze(0)
    ei_t = torch.tensor(ei, dtype=torch.long, device=device).unsqueeze(0)
    return nf_t, ei_t


# ─── Inference ──────────────────────────────────────────────────

@torch.no_grad()
def potential_denoise(model, config, manifold, instance, max_moves, n_steps,
                      device, temperature=1.0):
    """Denoise by sampling from potential-derived reverse distribution.

    Returns the TERMINAL state of the reverse trajectory (not best-seen).
    Also tracks best-seen for comparison.
    """
    model.eval()
    sol = manifold.sample_random(instance)

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

        # Model returns (state_potential, move_diffs)
        _, diffs = model(nf_t, ei_t, mn_t, mm_t, t_t)

        # Sample from reverse distribution
        temp = max(temperature * t_val, 0.1)
        n_use = len(moves)
        probs = F.softmax(diffs.squeeze(0)[:n_use] / temp, dim=-1)
        pick = torch.multinomial(probs, 1).item()

        # Apply unconditionally — true reverse diffusion
        sol = manifold.apply_move(sol, moves[pick])

    terminal_cost = manifold.cost(sol, instance)
    return sol, terminal_cost


@torch.no_grad()
def best_of_k_potential(model, config, manifold, instance, max_moves,
                        n_steps, device, K=8, temperature=1.0):
    """Run K reverse trajectories, return best terminal state."""
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
    model = StatePotentialModel(
        n_node_features=config.n_node_features,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(device)
    print(f"  StatePotentialModel: {sum(p.numel() for p in model.parameters()):,} params")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    n_pool = len(pool)
    best_gap = float('inf')
    n_denoise = t_max

    # Margin for contrastive losses
    margin = 1.0
    # Number of sampled moves for consistency loss
    n_consistency_moves = args.n_consistency_moves

    for epoch in range(1, args.n_epochs + 1):
        model.train()
        ep_loss, ep_l_contrast, ep_l_rank, ep_l_move, ep_l_consist = \
            0.0, 0.0, 0.0, 0.0, 0.0
        ep_acc = 0.0

        pbar = tqdm(range(args.steps_per_epoch), desc=f"  Epoch {epoch}")
        for step in pbar:
            # ── Collect batch ──
            batch_nf_xt, batch_ei_xt, batch_mn, batch_mm = [], [], [], []
            batch_nf_prev, batch_ei_prev = [], []
            batch_nf_neg, batch_ei_neg = [], []
            batch_t, batch_t_prev = [], []
            batch_inv_idx = []
            # For consistency: store (move, instance_idx) for K sampled moves
            consistency_data = []
            max_edges_xt, max_edges_other = 0, 0

            items = []
            for _ in range(args.batch_size):
                s = pool[np.random.randint(n_pool)]
                idx, x_t, x_prev = s['idx'], s['x_t'], s['x_prev']
                t, moves, inv_idx = s['t'], s['moves'], s['inv_idx']

                if not moves or inv_idx < 0:
                    continue

                t_norm = t / t_max
                t_prev_norm = max((t - 1) / t_max, 0.0)

                inst = instances[idx]

                # Features for x_t (with moves)
                nf_xt, ei_xt, mn, mm, _ = prepare_batch_item(
                    config, x_t, inst, moves, inv_idx, t_norm, args.max_moves)

                # Features for x_prev (state only, no moves)
                nf_prev = config.build_node_features(x_prev, inst, t_prev_norm)
                ei_prev = config.build_edges(x_prev, inst)

                # Features for x_neg (random tour, state only)
                x_neg = manifold.sample_random(inst)
                nf_neg = config.build_node_features(x_neg, inst, t_norm)
                ei_neg = config.build_edges(x_neg, inst)

                # Sample K moves for consistency loss
                n_moves = len(moves)
                if n_consistency_moves > 0 and n_moves > 0:
                    k_sample = min(n_consistency_moves, n_moves)
                    sampled_move_indices = np.random.choice(
                        n_moves, k_sample, replace=False)
                    sampled_moves = [moves[mi] for mi in sampled_move_indices]
                    # Apply each sampled move to get neighbor states
                    neighbor_states = []
                    for sm in sampled_moves:
                        y = manifold.apply_move(x_t, sm)
                        neighbor_states.append(y)
                    consistency_data.append({
                        'move_indices': sampled_move_indices,
                        'neighbors': neighbor_states,
                        'inst_idx': idx,
                        't_prev_norm': t_prev_norm,
                    })
                else:
                    consistency_data.append(None)

                items.append((nf_xt, ei_xt, mn, mm, nf_prev, ei_prev,
                              nf_neg, ei_neg, inv_idx, t_norm, t_prev_norm))
                max_edges_xt = max(max_edges_xt, ei_xt.shape[0])
                max_edges_other = max(max_edges_other,
                                      ei_prev.shape[0], ei_neg.shape[0])

            if not items:
                continue

            B = len(items)

            # Pad and stack
            for nf_xt, ei_xt, mn, mm, nf_prev, ei_prev, \
                nf_neg, ei_neg, inv_idx, t_n, t_p in items:
                # Pad edges for x_t
                if ei_xt.shape[0] < max_edges_xt:
                    pad = np.zeros((max_edges_xt - ei_xt.shape[0], 2), dtype=np.int64)
                    ei_xt = np.vstack([ei_xt, pad])
                batch_nf_xt.append(nf_xt)
                batch_ei_xt.append(ei_xt)
                batch_mn.append(mn)
                batch_mm.append(mm)

                # Pad edges for x_prev
                if ei_prev.shape[0] < max_edges_other:
                    pad = np.zeros((max_edges_other - ei_prev.shape[0], 2), dtype=np.int64)
                    ei_prev = np.vstack([ei_prev, pad])
                batch_nf_prev.append(nf_prev)
                batch_ei_prev.append(ei_prev)

                # Pad edges for x_neg
                if ei_neg.shape[0] < max_edges_other:
                    pad = np.zeros((max_edges_other - ei_neg.shape[0], 2), dtype=np.int64)
                    ei_neg = np.vstack([ei_neg, pad])
                batch_nf_neg.append(nf_neg)
                batch_ei_neg.append(ei_neg)

                batch_t.append(t_n)
                batch_t_prev.append(t_p)
                batch_inv_idx.append(inv_idx)

            # To tensors
            nf_xt_t = torch.tensor(np.stack(batch_nf_xt), dtype=torch.float32, device=device)
            ei_xt_t = torch.tensor(np.stack(batch_ei_xt), dtype=torch.long, device=device)
            mn_t = torch.tensor(np.stack(batch_mn), dtype=torch.long, device=device)
            mm_t = torch.tensor(np.stack(batch_mm), dtype=torch.bool, device=device)
            t_t = torch.tensor(batch_t, dtype=torch.float32, device=device)

            nf_prev_t = torch.tensor(np.stack(batch_nf_prev), dtype=torch.float32, device=device)
            ei_prev_t = torch.tensor(np.stack(batch_ei_prev), dtype=torch.long, device=device)
            t_prev_t = torch.tensor(batch_t_prev, dtype=torch.float32, device=device)

            nf_neg_t = torch.tensor(np.stack(batch_nf_neg), dtype=torch.float32, device=device)
            ei_neg_t = torch.tensor(np.stack(batch_ei_neg), dtype=torch.long, device=device)

            inv_idx_t = torch.tensor(batch_inv_idx, dtype=torch.long, device=device)

            # ── Forward passes ──

            # 1. x_t: get state potential + move diffs
            h_xt, f_xt = model.encode(nf_xt_t, ei_xt_t, t_t)
            diffs = model.move_diffs(h_xt, mn_t, mm_t)  # (B, M)

            # 2. x_prev: get state potential at t-1
            f_prev = model.forward_potential_only(nf_prev_t, ei_prev_t, t_prev_t)

            # 3. x_neg: get state potential at t (same time as x_t)
            f_neg = model.forward_potential_only(nf_neg_t, ei_neg_t, t_t)

            # ── Losses ──

            # Loss 1: Contrastive potential
            # f(x_t, t) should be higher than f(x_neg, t)
            # x_t is a sample from p_t (forward process), x_neg is random
            L_contrast = F.softplus(f_neg - f_xt + margin).mean()

            # Loss 2: Trajectory ranking
            # f(x_{t-1}, t-1) should be higher than f(x_t, t-1)
            # x_{t-1} is the predecessor: more likely at time t-1
            # We compute f(x_t, t-1) by re-encoding x_t with time t-1
            f_xt_at_prev = model.forward_potential_only(nf_xt_t, ei_xt_t, t_prev_t)
            L_rank = F.softplus(f_xt_at_prev - f_prev + margin).mean()

            # Loss 3: Move cross-entropy on inverse move
            # The inverse move should have the highest Δf
            L_move = F.cross_entropy(diffs, inv_idx_t)

            # Loss 4: Consistency — Δf(m) should match f(y_m, t-1) - f(x_t, t)
            # For K sampled moves per sample, compute actual potential difference
            L_consist = torch.tensor(0.0, device=device)
            n_consist_total = 0

            if n_consistency_moves > 0:
                for bi in range(B):
                    cd = consistency_data[bi]
                    if cd is None:
                        continue
                    inst = instances[cd['inst_idx']]
                    t_p = cd['t_prev_norm']
                    for ki, (mi, y_state) in enumerate(
                            zip(cd['move_indices'], cd['neighbors'])):
                        # Get predicted Δf for this move
                        pred_diff = diffs[bi, mi]

                        # Compute actual f(y, t-1)
                        nf_y = config.build_node_features(y_state, inst, t_p)
                        ei_y = config.build_edges(y_state, inst)
                        nf_y_t = torch.tensor(nf_y, dtype=torch.float32,
                                              device=device).unsqueeze(0)
                        ei_y_t = torch.tensor(ei_y, dtype=torch.long,
                                              device=device).unsqueeze(0)
                        t_y = torch.tensor([t_p], dtype=torch.float32, device=device)
                        with torch.no_grad():
                            f_y = model.forward_potential_only(nf_y_t, ei_y_t, t_y)

                        # Target: Δf(m) ≈ f(y, t-1) - f(x_t, t)  [stop-grad on target]
                        target_diff = (f_y - f_xt[bi]).detach()
                        L_consist = L_consist + (pred_diff - target_diff) ** 2
                        n_consist_total += 1

                if n_consist_total > 0:
                    L_consist = L_consist / n_consist_total

            # Total loss
            w_contrast = args.w_contrast
            w_rank = args.w_rank
            w_move = args.w_move
            w_consist = args.w_consist
            loss = (w_contrast * L_contrast + w_rank * L_rank +
                    w_move * L_move + w_consist * L_consist)

            # Accuracy: does the model predict the inverse move?
            acc = (diffs.argmax(dim=-1) == inv_idx_t).float().mean().item()

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item()
            ep_l_contrast += L_contrast.item()
            ep_l_rank += L_rank.item()
            ep_l_move += L_move.item()
            ep_l_consist += L_consist.item()
            ep_acc += acc
            pbar.set_postfix(loss=f"{loss.item():.3f}",
                             acc=f"{acc:.2%}",
                             ctr=f"{L_contrast.item():.2f}",
                             rnk=f"{L_rank.item():.2f}")

        scheduler.step()
        spe = args.steps_per_epoch
        print(f"  Epoch {epoch}: loss={ep_loss/spe:.4f} "
              f"[contrast={ep_l_contrast/spe:.3f} rank={ep_l_rank/spe:.3f} "
              f"move={ep_l_move/spe:.3f} consist={ep_l_consist/spe:.3f}] "
              f"acc={ep_acc/spe:.2%}")

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
                print(f"  [best] gap={eval_gap:+.2f}% -> {path}")

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
    parser.add_argument('--n_consistency_moves', type=int, default=4,
                        help='Number of sampled moves for consistency loss per sample')
    parser.add_argument('--w_contrast', type=float, default=1.0,
                        help='Weight for contrastive potential loss')
    parser.add_argument('--w_rank', type=float, default=1.0,
                        help='Weight for trajectory ranking loss')
    parser.add_argument('--w_move', type=float, default=1.0,
                        help='Weight for inverse move cross-entropy loss')
    parser.add_argument('--w_consist', type=float, default=0.5,
                        help='Weight for move consistency loss')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/potential')
    args = parser.parse_args()
    train(args)
