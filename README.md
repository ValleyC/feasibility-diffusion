# Feasibility Manifold Diffusion (FMD)

**Diffusion on the Feasibility Manifold for Combinatorial Optimization**

Official code for the paper: *Feasibility Manifold Diffusion: Annealed Score Matching on Structure-Preserving Solution Spaces for Combinatorial Optimization*.

## Overview

FMD defines a continuous-time Markov chain (CTMC) diffusion process exclusively over **feasible solutions**, where every intermediate state is a valid solution by construction. A GNN learns the Boltzmann score function at each temperature level, replacing expensive move-delta computation with a single forward pass at inference.

Key properties:
- **Guaranteed feasibility**: every state at every timestep satisfies all problem constraints
- **Label-free training**: no optimal solutions needed; self-play bootstraps quality
- **Surrogate delta**: the trained model replaces O(M) per-move cost evaluations with one GNN pass
- **Stochastic diversity**: temperature-controlled sampling produces diverse solutions; best-of-K improves with K

## Supported Problems (11 COPs)

| Problem | Manifold | Moves | State |
|---------|----------|-------|-------|
| TSP | Hamiltonian cycles | 2-opt | Tour permutation |
| ATSP | Directed Hamiltonian cycles | 2-opt | Tour permutation |
| CVRP | Capacity-respecting partitions | Relocate + swap | Assignment vector |
| CVRPTW | CVRP with time windows | Relocate + swap | Assignment vector |
| OVRP | Open VRP | Relocate + swap | Assignment vector |
| mTSP | Multi-agent TSP | Relocate + swap | Assignment vector |
| PCTSP | Prize-collecting TSP | Add/remove/swap | Node selection |
| SPCTSP | Stochastic PCTSP | Add/remove/swap | Node selection |
| OP | Orienteering problem | Add/remove/swap | Node selection |
| KP | Knapsack problem | Add/remove/swap | Item selection |
| MIS | Maximum independent set | Add/swap | Node selection |

## Project Structure

```
fmd/
├── core/
│   ├── manifold.py            # Abstract FeasibilityManifold interface
│   └── forward.py             # CTMC forward process
├── problems/
│   ├── tsp/                   # TSP: tour, manifold, 2-opt, k-NN moves
│   ├── cvrp/                  # CVRP: partition, relocate/swap
│   ├── cvrptw/                # CVRP with time windows
│   ├── ovrp/                  # Open VRP
│   ├── mtsp/                  # Multi-agent TSP
│   ├── atsp/                  # Asymmetric TSP
│   ├── pctsp/                 # Prize-collecting TSP
│   ├── spctsp/                # Stochastic PCTSP
│   ├── op/                    # Orienteering problem
│   ├── kp/                    # Knapsack problem
│   └── mis/                   # Maximum independent set
├── models/
│   └── problem_configs.py     # Per-problem feature extraction
├── training/
│   ├── generic_trainer.py     # FMD trainer (any problem)
│   └── generic_selfplay.py    # Self-play with best-of-K sampling
├── solvers/
│   └── batched_subtsp.py      # Sub-TSP solver for CVRP family
├── eval/
│   ├── scale_test.py          # Scale generalization evaluation
│   └── visualize.py           # Denoising visualization
└── tests/
    ├── test_all_manifolds.py  # All 11 manifold correctness tests
    ├── test_tsp_manifold.py   # TSP-specific tests
    ├── test_cvrp_partition.py # CVRP partition tests
    ├── test_pctsp.py          # PCTSP tests
    └── test_fmd_formulation.py # FMD smoke test (correlation, diversity, time conditioning)
```

## Quick Start

### Requirements

```bash
pip install numpy torch scipy tqdm
```

### Run Tests

```bash
# Verify all 11 manifolds
python -m tests.test_all_manifolds

# FMD formulation smoke test (TSP-20, ~3 min on CPU)
python -m tests.test_fmd_formulation
```

### Train

```bash
# TSP-50
python -m training.generic_selfplay \
    --problem tsp --N 50 --n_instances 5000 --device cuda:0 \
    --hidden_dim 128 --n_layers 6 --K 8

# CVRP-50
python -m training.generic_selfplay \
    --problem cvrp --N 50 --n_instances 5000 --device cuda:0 \
    --hidden_dim 128 --n_layers 6 --K 8
```

## Method

### Feasibility Manifold

For a CO instance, the feasibility manifold is a graph G = (F, E) where:
- F: set of all feasible solutions
- (x, y) in E iff y is reachable from x by one feasibility-preserving move

### Annealed Score Matching

The model learns the discrete Boltzmann score at each temperature level:

```
s(x, y, t) = exp(-delta_c(x -> y) / tau_t)
```

Training target: `softmax(-delta / tau_t)` over all feasible moves.

### Inference (Reverse Sampling)

```
1. x_T <- random feasible solution
2. for t = T to 0:
     moves <- enumerate_moves(x_t)      # all feasible neighbors
     scores <- GNN(x_t, moves, t)       # one forward pass
     m* ~ Categorical(softmax(scores))  # stochastic sampling
     if move_delta(m*) < 0:
       x_{t-1} <- apply(x_t, m*)
3. return x_0
```

Best-of-K: run K stochastic trajectories, return the best solution.

## License

MIT License. See [LICENSE](LICENSE).
