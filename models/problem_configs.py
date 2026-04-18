"""
Problem-specific configurations for the generic trainer.

Each config defines:
  - build_node_features: how to encode the current state as per-node features
  - build_edges: which graph structure to give the GNN
  - move_to_4nodes: which 4 nodes characterize each move (for scoring)
  - create_instance: how to generate random instances

The GNN and move scorer MLP are SHARED across problems.
Only the feature extraction is problem-specific — this ensures
no performance sacrifice while keeping the framework unified.
"""

import numpy as np
from abc import ABC, abstractmethod
from problems.tsp.tour import dist_matrix_from_coords


class ProblemConfig(ABC):
    """Base class for problem-specific feature extraction."""

    name: str = ""
    n_node_features: int = 0

    @abstractmethod
    def create_instance(self, N: int, seed: int) -> dict:
        """Generate a random instance."""

    @abstractmethod
    def create_manifold(self):
        """Return the appropriate FeasibilityManifold."""

    @abstractmethod
    def build_node_features(self, solution, instance, progress: float) -> np.ndarray:
        """Build per-node feature matrix. Returns (n_nodes, n_node_features)."""

    @abstractmethod
    def build_edges(self, solution, instance) -> np.ndarray:
        """Build edge index for GNN. Returns (E, 2) int64."""

    @abstractmethod
    def move_to_4nodes(self, solution, move, instance) -> tuple:
        """Extract 4 critical node indices for move scoring."""

    def n_nodes(self, instance) -> int:
        """Number of nodes in the GNN graph."""
        raise NotImplementedError


# ─── TSP ─────────────────────────────────────────────────────

class TSPConfig(ProblemConfig):
    name = "TSP"
    n_node_features = 5  # x, y, tour_pos, sin_t, cos_t

    def create_instance(self, N, seed):
        from problems.tsp.data import generate_instance
        coords = generate_instance(N, seed=seed)
        dist = dist_matrix_from_coords(coords)
        return {'coords': coords, 'dist': dist, 'N': N}

    def create_manifold(self):
        from problems.tsp.manifold import TSPManifold
        return TSPManifold()

    def n_nodes(self, instance):
        return instance['N']

    def build_node_features(self, solution, instance, progress):
        coords = instance['coords']
        N = len(solution)
        pos = np.zeros(N, dtype=np.float32)
        for p, node in enumerate(solution):
            pos[node] = p / N
        feats = np.zeros((N, 5), dtype=np.float32)
        feats[:, 0:2] = coords
        feats[:, 2] = pos
        feats[:, 3] = np.sin(2 * np.pi * progress)
        feats[:, 4] = np.cos(2 * np.pi * progress)
        return feats

    def build_edges(self, solution, instance):
        N = len(solution)
        edges = []
        for i in range(N):
            a, b = solution[i], solution[(i + 1) % N]
            edges.append([a, b])
            edges.append([b, a])
        return np.array(edges, dtype=np.int64)

    def move_to_4nodes(self, solution, move, instance):
        i, j = move
        N = len(solution)
        return (solution[i], solution[i + 1],
                solution[j], solution[(j + 1) % N])


# ─── CVRP (Partition) ───────────────────────────────────────

class CVRPConfig(ProblemConfig):
    name = "CVRP"
    n_node_features = 6  # x, y, demand/cap, vehicle_id/K, is_depot, progress

    def create_instance(self, N, seed):
        np.random.seed(seed)
        coords = np.random.rand(N + 1, 2).astype(np.float32)
        demands = np.zeros(N + 1, dtype=np.float32)
        demands[1:] = np.random.randint(1, 10, N).astype(np.float32)
        cap = {10: 20, 20: 30, 50: 40, 100: 50}.get(N, 50)
        dist = dist_matrix_from_coords(coords)
        return {'coords': coords, 'demands': demands, 'capacity': float(cap),
                'dist': dist, 'n_customers': N}

    def create_manifold(self):
        from problems.cvrp.partition_manifold import CVRPPartitionManifold
        return CVRPPartitionManifold()

    def n_nodes(self, instance):
        return instance['n_customers'] + 1

    def build_node_features(self, solution, instance, progress):
        N = instance['n_customers']
        coords = instance['coords']
        demands = instance['demands']
        cap = instance['capacity']
        K = max(solution[1:].max() + 1, 1)

        feats = np.zeros((N + 1, 6), dtype=np.float32)
        feats[:, 0:2] = coords
        feats[:, 2] = demands / cap
        feats[:, 3] = np.where(solution >= 0, solution / K, 0)
        feats[0, 4] = 1.0  # is_depot
        feats[:, 5] = progress
        return feats

    def build_edges(self, solution, instance):
        # k-NN spatial edges (k=10) — works for any partition state
        coords = instance['coords']
        N = len(coords)
        k = min(10, N - 1)
        dists = np.linalg.norm(coords[:, None] - coords[None], axis=-1)
        np.fill_diagonal(dists, np.inf)
        edges = []
        for i in range(N):
            neighbors = np.argsort(dists[i])[:k]
            for j in neighbors:
                edges.append([i, j])
        return np.array(edges, dtype=np.int64) if edges else np.zeros((0, 2), dtype=np.int64)

    def move_to_4nodes(self, solution, move, instance):
        from problems.cvrp.partition import PartMoveType, get_vehicle_customers
        mtype = move[0]
        if mtype == PartMoveType.RELOCATE:
            customer, k_src, k_dst = move[1], move[2], move[3]
            # Context: customer, depot, a customer in dst vehicle, depot
            dst_custs = get_vehicle_customers(solution, k_dst)
            ctx = dst_custs[0] if dst_custs else 0
            return (customer, 0, ctx, 0)
        elif mtype == PartMoveType.SWAP:
            a, _, b, _ = move[1], move[2], move[3], move[4]
            return (a, 0, b, 0)
        return (0, 0, 0, 0)


# ─── PCTSP ───────────────────────────────────────────────────

class PCTSPConfig(ProblemConfig):
    name = "PCTSP"
    n_node_features = 7  # x, y, prize, penalty, is_selected, is_depot, progress

    def create_instance(self, N, seed):
        np.random.seed(seed)
        coords = np.random.rand(N + 1, 2).astype(np.float32)
        prizes = np.zeros(N + 1, dtype=np.float32)
        prizes[1:] = np.random.uniform(0.1, 1.0, N).astype(np.float32)
        penalties = np.zeros(N + 1, dtype=np.float32)
        penalties[1:] = np.random.uniform(0.1, 0.5, N).astype(np.float32)
        dist = dist_matrix_from_coords(coords)
        return {'coords': coords, 'prizes': prizes, 'penalties': penalties,
                'min_prize': float(prizes.sum() * 0.5), 'dist': dist, 'n_customers': N}

    def create_manifold(self):
        from problems.pctsp.manifold import PCTSPManifold
        return PCTSPManifold()

    def n_nodes(self, instance):
        return instance['n_customers'] + 1

    def build_node_features(self, solution, instance, progress):
        N = instance['n_customers']
        feats = np.zeros((N + 1, 7), dtype=np.float32)
        feats[:, 0:2] = instance['coords']
        feats[:, 2] = instance['prizes']
        feats[:, 3] = instance['penalties']
        feats[:, 4] = solution.astype(np.float32)
        feats[0, 5] = 1.0
        feats[:, 6] = progress
        return feats

    def build_edges(self, solution, instance):
        return CVRPConfig.build_edges(self, solution, instance)

    def move_to_4nodes(self, solution, move, instance):
        mtype, a, b = move
        # Context: the node(s) involved + nearest selected neighbors
        selected = np.where(solution)[0]
        if len(selected) > 1:
            dists = instance['dist'][a, selected]
            nearest = selected[np.argsort(dists)[1]] if a in selected else selected[np.argsort(dists)[0]]
        else:
            nearest = 0
        if b >= 0:
            return (a, nearest, b, 0)
        return (a, nearest, 0, 0)


# ─── OP ──────────────────────────────────────────────────────

class OPConfig(ProblemConfig):
    name = "OP"
    n_node_features = 6  # x, y, prize, is_selected, is_depot, progress

    def create_instance(self, N, seed):
        np.random.seed(seed)
        coords = np.random.rand(N + 1, 2).astype(np.float32)
        prizes = np.zeros(N + 1, dtype=np.float32)
        prizes[1:] = np.random.uniform(0.5, 2.0, N).astype(np.float32)
        dist = dist_matrix_from_coords(coords)
        return {'coords': coords, 'prizes': prizes, 'budget': 4.0,
                'dist': dist, 'n_customers': N}

    def create_manifold(self):
        from problems.op.manifold import OPManifold
        return OPManifold()

    def n_nodes(self, instance):
        return instance['n_customers'] + 1

    def build_node_features(self, solution, instance, progress):
        N = instance['n_customers']
        feats = np.zeros((N + 1, 6), dtype=np.float32)
        feats[:, 0:2] = instance['coords']
        feats[:, 2] = instance['prizes']
        feats[:, 3] = solution.astype(np.float32)
        feats[0, 4] = 1.0
        feats[:, 5] = progress
        return feats

    def build_edges(self, solution, instance):
        return CVRPConfig.build_edges(self, solution, instance)

    def move_to_4nodes(self, solution, move, instance):
        return PCTSPConfig.move_to_4nodes(self, solution, move, instance)


# ─── KP (non-geometric) ─────────────────────────────────────

class KPConfig(ProblemConfig):
    name = "KP"
    n_node_features = 5  # value, weight, v/w ratio, is_selected, progress

    def create_instance(self, N, seed):
        np.random.seed(seed)
        values = np.random.uniform(1, 10, N).astype(np.float32)
        weights = np.random.uniform(1, 5, N).astype(np.float32)
        return {'values': values, 'weights': weights,
                'capacity': float(weights.sum() * 0.4), 'n_items': N}

    def create_manifold(self):
        from problems.kp.manifold import KPManifold
        return KPManifold()

    def n_nodes(self, instance):
        return instance['n_items']

    def build_node_features(self, solution, instance, progress):
        N = instance['n_items']
        vals = instance['values']
        wts = instance['weights']
        feats = np.zeros((N, 5), dtype=np.float32)
        feats[:, 0] = vals / vals.max()
        feats[:, 1] = wts / instance['capacity']
        feats[:, 2] = (vals / wts) / (vals / wts).max()
        feats[:, 3] = solution.astype(np.float32)
        feats[:, 4] = progress
        return feats

    def build_edges(self, solution, instance):
        # Fully connected for small KP; k-NN by value/weight ratio for large
        N = instance['n_items']
        if N <= 50:
            edges = [[i, j] for i in range(N) for j in range(N) if i != j]
        else:
            vw = instance['values'] / instance['weights']
            dists = np.abs(vw[:, None] - vw[None, :])
            k = min(10, N - 1)
            edges = []
            for i in range(N):
                neighbors = np.argsort(dists[i])[1:k + 1]
                for j in neighbors:
                    edges.append([i, j])
        return np.array(edges, dtype=np.int64)

    def move_to_4nodes(self, solution, move, instance):
        mtype, a, b = move
        # For KP: the item + items with similar value/weight ratio
        vw = instance['values'] / instance['weights']
        order = np.argsort(np.abs(vw - vw[a]))
        ctx1 = order[1] if len(order) > 1 else a
        ctx2 = order[2] if len(order) > 2 else a
        if b >= 0:
            return (a, ctx1, b, ctx2)
        return (a, ctx1, ctx2, ctx2)


# ─── MIS (non-geometric) ────────────────────────────────────

class MISConfig(ProblemConfig):
    name = "MIS"
    n_node_features = 5  # degree_norm, is_selected, n_sel_neighbors, conflict, progress

    def create_instance(self, N, seed):
        np.random.seed(seed)
        adj = np.random.random((N, N)) < 0.15
        adj = adj | adj.T
        np.fill_diagonal(adj, False)
        return {'adj': adj, 'n_nodes': N}

    def create_manifold(self):
        from problems.mis.manifold import MISManifold
        return MISManifold()

    def n_nodes(self, instance):
        return instance['n_nodes']

    def build_node_features(self, solution, instance, progress):
        N = instance['n_nodes']
        adj = instance['adj']
        degree = adj.sum(axis=1).astype(np.float32)
        max_deg = max(degree.max(), 1)
        n_sel_neighbors = (adj & solution[None, :]).sum(axis=1).astype(np.float32)
        conflict = (n_sel_neighbors > 0).astype(np.float32)

        feats = np.zeros((N, 5), dtype=np.float32)
        feats[:, 0] = degree / max_deg
        feats[:, 1] = solution.astype(np.float32)
        feats[:, 2] = n_sel_neighbors / max(max_deg, 1)
        feats[:, 3] = conflict
        feats[:, 4] = progress
        return feats

    def build_edges(self, solution, instance):
        adj = instance['adj']
        edges = np.argwhere(adj)
        return edges.astype(np.int64) if len(edges) > 0 else np.zeros((0, 2), dtype=np.int64)

    def move_to_4nodes(self, solution, move, instance):
        mtype, a, b = move
        adj = instance['adj']
        neighbors_a = np.where(adj[a])[0]
        ctx1 = neighbors_a[0] if len(neighbors_a) > 0 else a
        ctx2 = neighbors_a[1] if len(neighbors_a) > 1 else ctx1
        if b >= 0:
            return (a, ctx1, b, ctx2)
        return (a, ctx1, ctx2, ctx2)


# ─── mTSP ────────────────────────────────────────────────────

class MTSPConfig(ProblemConfig):
    name = "mTSP"
    n_node_features = 5  # x, y, agent_id/K, is_depot, progress

    def create_instance(self, N, seed, K=3):
        np.random.seed(seed)
        coords = np.random.rand(N + 1, 2).astype(np.float32)
        dist = dist_matrix_from_coords(coords)
        return {'coords': coords, 'dist': dist, 'n_customers': N, 'n_agents': K}

    def create_manifold(self):
        from problems.mtsp.manifold import MTSPManifold
        return MTSPManifold()

    def n_nodes(self, instance):
        return instance['n_customers'] + 1

    def build_node_features(self, solution, instance, progress):
        N = instance['n_customers']
        K = instance['n_agents']
        feats = np.zeros((N + 1, 5), dtype=np.float32)
        feats[:, 0:2] = instance['coords']
        feats[:, 2] = np.where(solution >= 0, solution / K, 0)
        feats[0, 3] = 1.0
        feats[:, 4] = progress
        return feats

    def build_edges(self, solution, instance):
        return CVRPConfig.build_edges(self, solution, instance)

    def move_to_4nodes(self, solution, move, instance):
        mtype = move[0]
        if mtype == 0:  # RELOCATE
            customer, _, k_dst = move[1], move[2], move[3]
            dst_custs = [i for i in range(len(solution)) if solution[i] == k_dst]
            ctx = dst_custs[0] if dst_custs else 0
            return (customer, 0, ctx, 0)
        else:  # SWAP
            a, _, b, _ = move[1], move[2], move[3], move[4]
            return (a, 0, b, 0)


# ─── Registry ────────────────────────────────────────────────

PROBLEM_CONFIGS = {
    'tsp': TSPConfig,
    'cvrp': CVRPConfig,
    'pctsp': PCTSPConfig,
    'op': OPConfig,
    'kp': KPConfig,
    'mis': MISConfig,
    'mtsp': MTSPConfig,
}
