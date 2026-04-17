# Feasible Diffusion: Diffusion on the Feasibility Manifold for Combinatorial Optimization

A continuous-time diffusion framework where **every intermediate state is a valid solution**.

## Core Idea

Instead of diffusing over edge indicators or node labels (where most intermediate states are infeasible), we define a CTMC over the set of feasible solutions, with transitions given by structure-preserving local search moves.

- **Forward process**: random valid perturbations (e.g., 2-opt swaps) progressively scramble a good solution
- **Reverse process**: a learned neural network denoises back toward optimality
- **Every state at every timestep is a feasible solution** — no post-hoc projection or repair needed

## Supported Problems

| Problem | State Space | Transitions | Status |
|---------|------------|-------------|--------|
| TSP | Hamiltonian cycles | 2-opt swaps | ✅ Core implemented |
| CVRP | Feasible partitions | Customer swaps | 🔜 Planned |

## Project Structure

```
feasible-diffusion/
├── core/
│   ├── manifold.py       # Abstract feasibility manifold interface
│   └── forward.py        # CTMC forward process (scrambling)
├── problems/
│   └── tsp/
│       ├── tour.py       # Tour representation + 2-opt operations
│       ├── manifold.py   # TSP feasibility manifold
│       └── data.py       # Instance generation + 2-opt solver
├── models/               # Neural move scorer (TODO)
├── training/             # Training loops (TODO)
├── eval/                 # Evaluation + baselines (TODO)
└── tests/
    └── test_tsp_manifold.py  # Correctness + invariant tests
```

## Quick Start

```bash
# Run correctness tests
python tests/test_tsp_manifold.py
```

## Key Theoretical Properties

1. **Feasibility preservation**: every transition stays on the feasibility manifold (verified by tests)
2. **Ergodicity**: 2-opt neighborhood graph is connected (Lin, 1965), ensuring CTMC mixes to uniform
3. **O(1) move evaluation**: cost change from any 2-opt swap computed in constant time
4. **E(2)-invariance**: tour cost and 2-opt deltas are invariant under rotation/translation/reflection
