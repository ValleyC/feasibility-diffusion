"""
Diagnose multiprocessing issues on this machine.

Tests:
  1. Basic multiprocessing works
  2. Pickling of manifold class works
  3. _process_one_instance works in a worker
  4. Pool with spawn context works
  5. Pool with fork context works
  6. Full pool generation works (small scale)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import numpy as np
import multiprocessing as mp
from multiprocessing import cpu_count
import pickle


def test_basic_mp():
    """Test 1: basic multiprocessing works at all."""
    def square(x):
        return x * x

    print(f"CPUs: {cpu_count()}")

    for method in ['fork', 'spawn']:
        try:
            ctx = mp.get_context(method)
            with ctx.Pool(2) as p:
                result = p.map(square, [1, 2, 3, 4])
            assert result == [1, 4, 9, 16]
            print(f"  [{method}] basic Pool.map: PASS")
        except Exception as e:
            print(f"  [{method}] basic Pool.map: FAIL — {e}")


def test_pickle_manifold():
    """Test 2: can we pickle the manifold class?"""
    from problems.cvrp.partition_manifold import CVRPPartitionManifold

    try:
        data = pickle.dumps(CVRPPartitionManifold)
        cls = pickle.loads(data)
        m = cls()
        print(f"  Pickle CVRPPartitionManifold class: PASS")
    except Exception as e:
        print(f"  Pickle CVRPPartitionManifold class: FAIL — {e}")

    try:
        inst = CVRPPartitionManifold()
        data = pickle.dumps(inst)
        obj = pickle.loads(data)
        print(f"  Pickle CVRPPartitionManifold instance: PASS")
    except Exception as e:
        print(f"  Pickle CVRPPartitionManifold instance: FAIL — {e}")


def test_pickle_instance():
    """Test 3: can we pickle a CVRP instance dict?"""
    from models.problem_configs import CVRPConfig
    config = CVRPConfig()
    inst = config.create_instance(20, seed=42)

    try:
        data = pickle.dumps(inst)
        obj = pickle.loads(data)
        print(f"  Pickle CVRP instance: PASS (size={len(data)} bytes)")
    except Exception as e:
        print(f"  Pickle CVRP instance: FAIL — {e}")


def test_pickle_work_args():
    """Test 4: can we pickle the full work_args tuple?"""
    from problems.cvrp.partition_manifold import CVRPPartitionManifold
    from models.problem_configs import CVRPConfig

    config = CVRPConfig()
    inst = config.create_instance(20, seed=42)
    manifold_class = CVRPPartitionManifold

    work_args = (0, inst, manifold_class, 500, 300, 3)

    try:
        data = pickle.dumps(work_args)
        obj = pickle.loads(data)
        print(f"  Pickle work_args: PASS (size={len(data)} bytes)")
    except Exception as e:
        print(f"  Pickle work_args: FAIL — {e}")


def test_worker_function():
    """Test 5: does _process_one_instance work directly?"""
    from training.generic_trainer import _process_one_instance
    from problems.cvrp.partition_manifold import CVRPPartitionManifold
    from models.problem_configs import CVRPConfig

    config = CVRPConfig()
    inst = config.create_instance(20, seed=42)

    args = (0, inst, CVRPPartitionManifold, 500, 50, 1)  # 1 restart, 50 max iters

    t0 = time.time()
    try:
        samples, cost = _process_one_instance(args)
        elapsed = time.time() - t0
        print(f"  _process_one_instance (N=20): PASS "
              f"({len(samples)} samples, cost={cost:.4f}, {elapsed:.2f}s)")
    except Exception as e:
        print(f"  _process_one_instance: FAIL — {e}")


def test_pool_with_worker():
    """Test 6: can Pool run _process_one_instance?"""
    from training.generic_trainer import _process_one_instance
    from problems.cvrp.partition_manifold import CVRPPartitionManifold
    from models.problem_configs import CVRPConfig

    config = CVRPConfig()
    instances = [config.create_instance(20, seed=42 + i) for i in range(4)]

    work_args = [
        (i, inst, CVRPPartitionManifold, 500, 50, 1)
        for i, inst in enumerate(instances)
    ]

    for method in ['fork', 'spawn']:
        try:
            ctx = mp.get_context(method)
            t0 = time.time()
            with ctx.Pool(2) as p:
                results = p.map(_process_one_instance, work_args, chunksize=1)
            elapsed = time.time() - t0
            total_samples = sum(len(r[0]) for r in results)
            print(f"  [{method}] Pool._process_one_instance (4 instances): PASS "
                  f"({total_samples} samples, {elapsed:.2f}s)")
        except Exception as e:
            print(f"  [{method}] Pool._process_one_instance: FAIL — {e}")


def test_pool_larger():
    """Test 7: larger scale pool test."""
    from training.generic_trainer import _process_one_instance
    from problems.cvrp.partition_manifold import CVRPPartitionManifold
    from models.problem_configs import CVRPConfig

    config = CVRPConfig()
    N_INST = 20
    instances = [config.create_instance(20, seed=42 + i) for i in range(N_INST)]

    work_args = [
        (i, inst, CVRPPartitionManifold, 500, 50, 1)
        for i, inst in enumerate(instances)
    ]

    n_workers = min(8, cpu_count())

    for method in ['fork', 'spawn']:
        try:
            ctx = mp.get_context(method)
            t0 = time.time()
            with ctx.Pool(n_workers) as p:
                results = list(p.imap(_process_one_instance, work_args, chunksize=2))
            elapsed = time.time() - t0
            total_samples = sum(len(r[0]) for r in results)
            costs = [r[1] for r in results]
            print(f"  [{method}] imap {N_INST} instances, {n_workers} workers: PASS "
                  f"({total_samples} samples, avg cost={np.mean(costs):.4f}, {elapsed:.2f}s)")
        except Exception as e:
            print(f"  [{method}] imap {N_INST} instances: FAIL — {e}")


def test_cvrp50_single():
    """Test 8: single CVRP-50 instance timing."""
    from training.generic_trainer import _process_one_instance
    from problems.cvrp.partition_manifold import CVRPPartitionManifold
    from models.problem_configs import CVRPConfig

    config = CVRPConfig()
    inst = config.create_instance(50, seed=42)

    args = (0, inst, CVRPPartitionManifold, 2000, 100, 1)

    t0 = time.time()
    try:
        samples, cost = _process_one_instance(args)
        elapsed = time.time() - t0
        print(f"  CVRP-50 single instance: PASS "
              f"({len(samples)} samples, cost={cost:.4f}, {elapsed:.1f}s)")
    except Exception as e:
        print(f"  CVRP-50 single: FAIL — {e}")


if __name__ == '__main__':
    print("=" * 60)
    print("Multiprocessing Diagnostics")
    print("=" * 60)

    print("\n1. Basic multiprocessing:")
    test_basic_mp()

    print("\n2. Pickle manifold:")
    test_pickle_manifold()

    print("\n3. Pickle instance:")
    test_pickle_instance()

    print("\n4. Pickle work_args:")
    test_pickle_work_args()

    print("\n5. Worker function (direct):")
    test_worker_function()

    print("\n6. Pool with worker (4 instances):")
    test_pool_with_worker()

    print("\n7. Pool larger (20 instances, imap):")
    test_pool_larger()

    print("\n8. CVRP-50 single instance timing:")
    test_cvrp50_single()

    print("\n" + "=" * 60)
    print("Diagnostics complete.")
