"""
Visualize the denoising process as a GIF.

Each frame shows a valid Hamiltonian cycle. The tour improves step by step
from a random starting tour toward a near-optimal solution — and EVERY
intermediate state is a feasible tour.

Usage:
    cd feasibility-diffusion
    python -m eval.visualize \
        --ckpt checkpoints/clean/best_tsp50.pt \
        --N 50 --n_steps 150 --device cuda:0 \
        --output denoise_tsp50.gif
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import imageio.v2 as imageio

from problems.tsp.tour import (
    enumerate_2opt, apply_2opt, delta_2opt, tour_cost,
    dist_matrix_from_coords, random_tour, is_valid_tour,
)
from problems.tsp.data import generate_instance, solve_2opt
from models.move_scorer import MoveScorer


def run_denoising_with_trajectory(model, coords, dist, moves_ij, N,
                                  n_steps, device):
    """Run greedy denoising and record the tour at every step.

    Returns:
        tours: list of (N,) numpy arrays — tour at each step
        costs: list of floats — cost at each step
    """
    model.eval()
    moves_list = moves_ij.cpu().numpy().tolist()
    coords_t = torch.tensor(coords, dtype=torch.float32, device=device).unsqueeze(0)

    tour_np = random_tour(N)
    tours = [tour_np.copy()]
    costs = [tour_cost(tour_np, dist)]

    with torch.no_grad():
        for step in range(n_steps):
            progress = step / max(n_steps - 1, 1)
            tour_t = torch.tensor(tour_np, dtype=torch.long, device=device).unsqueeze(0)
            t_tensor = torch.tensor([progress], device=device)

            scores = model(coords_t, tour_t, t_tensor, 1.0, moves_ij)
            best_idx = scores.argmax(dim=-1).item()
            i, j = moves_list[best_idx]

            d = delta_2opt(tour_np, i, j, dist)
            if d < 0:
                tour_np = apply_2opt(tour_np, i, j)

            tours.append(tour_np.copy())
            costs.append(tour_cost(tour_np, dist))

    return tours, costs


def draw_tour(ax, coords, tour, cost, step, total_steps, opt_cost=None,
              title_prefix=""):
    """Draw a single tour on a matplotlib axis."""
    ax.clear()
    N = len(tour)

    # Draw edges
    segments = []
    for k in range(N):
        p1 = coords[tour[k]]
        p2 = coords[tour[(k + 1) % N]]
        segments.append([p1, p2])

    lc = LineCollection(segments, colors='#2196F3', linewidths=1.5, alpha=0.8)
    ax.add_collection(lc)

    # Draw nodes
    ax.scatter(coords[:, 0], coords[:, 1], c='#333333', s=30, zorder=5)

    # Axis settings
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

    # Title
    gap_str = ""
    if opt_cost is not None and opt_cost > 0:
        gap = (cost / opt_cost - 1) * 100
        gap_str = f"  |  gap: {gap:.1f}%"

    ax.set_title(
        f"{title_prefix}Step {step}/{total_steps}  |  cost: {cost:.4f}{gap_str}",
        fontsize=12, fontweight='bold',
    )

    # Feasibility badge
    ax.text(0.98, 0.02, "FEASIBLE", transform=ax.transAxes,
            fontsize=8, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#4CAF50',
                      edgecolor='none', alpha=0.9),
            color='white', fontweight='bold')


def make_gif(coords, tours, costs, output_path, opt_cost=None,
             fps=8, title_prefix=""):
    """Generate a GIF from a sequence of tours."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
    total_steps = len(tours) - 1
    frames = []

    # Select frames (don't render all 150 — pick key moments + regular samples)
    n_tours = len(tours)
    if n_tours <= 60:
        frame_indices = list(range(n_tours))
    else:
        # First 20 steps (fast early improvement), then every 5th
        early = list(range(min(20, n_tours)))
        late = list(range(20, n_tours, 5))
        frame_indices = sorted(set(early + late + [n_tours - 1]))

    for idx in frame_indices:
        draw_tour(ax, coords, tours[idx], costs[idx], idx, total_steps,
                  opt_cost=opt_cost, title_prefix=title_prefix)
        fig.tight_layout()

        # Render to image
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()
        image = np.asarray(buf)[:, :, :3].copy()  # RGBA -> RGB
        frames.append(image)

    # Add extra copies of last frame (pause at end)
    for _ in range(fps * 2):
        frames.append(frames[-1])

    plt.close(fig)

    imageio.mimsave(output_path, frames, fps=fps, loop=0)
    print(f"Saved GIF: {output_path} ({len(frames)} frames, {fps} fps)")


def make_cost_plot(costs, output_path, opt_cost=None):
    """Plot cost vs denoising step."""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(costs, color='#2196F3', linewidth=2, label='Tour cost')
    if opt_cost is not None:
        ax.axhline(y=opt_cost, color='#4CAF50', linestyle='--',
                    linewidth=1.5, label=f'2-opt reference ({opt_cost:.4f})')
    ax.set_xlabel('Denoising step', fontsize=12)
    ax.set_ylabel('Tour cost', fontsize=12)
    ax.set_title('Feasibility-Preserving Denoising: Cost vs Step', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved cost plot: {output_path}")


def main(args):
    device = torch.device(args.device)
    N = args.N

    moves_list = enumerate_2opt(N)
    M = len(moves_list)
    moves_ij = torch.tensor(moves_list, dtype=torch.long, device=device)

    # Load checkpoint
    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    ckpt_args = ckpt.get('args', {})

    model = MoveScorer(
        n_node_features=5,
        hidden_dim=ckpt_args.get('hidden_dim', 128),
        n_layers=ckpt_args.get('n_layers', 6),
    ).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"  Model loaded (gap={ckpt.get('gap', '?')}, epoch={ckpt.get('epoch', '?')})")

    # Generate instances
    np.random.seed(args.seed)
    for inst_id in range(args.n_instances):
        coords = generate_instance(N, seed=args.seed + inst_id)
        dist = dist_matrix_from_coords(coords)

        # Reference: 2-opt solution
        ref_tour, ref_cost = solve_2opt(coords, max_restarts=5)

        print(f"\nInstance {inst_id}: 2-opt ref cost = {ref_cost:.4f}")

        # Run denoising
        tours, costs = run_denoising_with_trajectory(
            model, coords, dist, moves_ij, N, args.n_steps, device,
        )
        print(f"  Denoised: {costs[0]:.4f} -> {costs[-1]:.4f} "
              f"(gap: {(costs[-1]/ref_cost - 1)*100:.1f}%)")

        # Verify every tour is feasible
        for step, t in enumerate(tours):
            assert is_valid_tour(t), f"Step {step}: INFEASIBLE TOUR!"
        print(f"  All {len(tours)} intermediate tours are FEASIBLE")

        # Generate outputs
        base = os.path.splitext(args.output)[0]
        gif_path = f"{base}_inst{inst_id}.gif"
        plot_path = f"{base}_inst{inst_id}_cost.png"

        make_gif(coords, tours, costs, gif_path, opt_cost=ref_cost,
                 fps=args.fps, title_prefix=f"TSP-{N} ")
        make_cost_plot(costs, plot_path, opt_cost=ref_cost)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--N', type=int, default=50)
    parser.add_argument('--n_steps', type=int, default=150, help='Denoising steps')
    parser.add_argument('--n_instances', type=int, default=3, help='Number of instances to visualize')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--fps', type=int, default=8)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--output', type=str, default='denoise_tsp50.gif')
    args = parser.parse_args()
    main(args)
