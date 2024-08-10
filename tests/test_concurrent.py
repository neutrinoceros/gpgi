# exercise some thread concurrency
# https://py-free-threading.github.io/debugging/


import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import gpgi
from gpgi._boundaries import BoundaryRegistry

prng = np.random.default_rng()


N_THREADS = 10


def random_dataset():
    return gpgi.load(
        geometry="cartesian",
        grid={
            "cell_edges": {
                "x": np.linspace(-0.1, 1.1, 13),
                "y": np.linspace(-0.1, 1.1, 13),
            },
        },
        particles={
            "coordinates": {
                "x": prng.random(10_000),
                "y": prng.random(10_000),
            }
        },
    )


class TestSetupHostCellIndex:
    def check(self, results):
        # Check results: verify that all threads see the same array
        assert len(set(results)) == 1
        # check that we indeed collected ids, not None
        assert isinstance(results[0], int)

    def test_concurrent_threading(self):
        # Defines a thread barrier that will be spawned before parallel execution
        # this increases the probability of concurrent access clashes.
        barrier = threading.Barrier(N_THREADS)

        # This object will be shared by all the threads.
        ds = random_dataset()

        results = []

        def closure():
            # Ensure that all threads reach this point before concurrent execution.
            barrier.wait()
            hci = ds._setup_host_cell_index()
            assert ds.host_cell_index is hci
            results.append(id(hci))

        # Spawn n threads that call _setup_host_cell_index concurrently.
        workers = []
        for _ in range(0, N_THREADS):
            workers.append(threading.Thread(target=closure))

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()

        self.check(results)

    def test_concurrent_pool(self):
        # Defines a thread barrier that will be spawned before parallel execution
        # this increases the probability of concurrent access clashes.
        barrier = threading.Barrier(N_THREADS)

        # This object will be shared by all the threads.
        ds = random_dataset()

        def closure():
            # Ensure that all threads reach this point before concurrent execution.
            barrier.wait()
            hci = ds._setup_host_cell_index()
            assert ds.host_cell_index is hci
            return id(hci)

        with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
            futures = [executor.submit(closure) for _ in range(N_THREADS)]

        results = [f.result() for f in futures]
        self.check(results)


class TestBoundaryRegistry:
    def check(self, results):
        # Check results: verify that all threads get the same obj
        assert len(set(results)) == 1

    def test_concurrent_threading(self):
        # Defines a thread barrier that will be spawned before parallel execution
        # this increases the probability of concurrent access clashes.
        barrier = threading.Barrier(N_THREADS)

        # This object will be shared by all the threads.
        registry = BoundaryRegistry()

        def test_recipe(
            same_side_active_layer,
            same_side_ghost_layer,
            opposite_side_active_layer,
            opposite_side_ghost_layer,
            weight_same_side_active_layer,
            weight_same_side_ghost_layer,
            weight_opposite_side_active_layer,
            weight_opposite_side_ghost_layer,
            side,
            metadata,
        ): ...

        results = []

        def closure():
            assert "test" not in registry

            # Ensure that all threads reach this point before concurrent execution.
            barrier.wait()
            registry.register("test", test_recipe)
            results.append(registry["test"])

        # Spawn n threads that call _setup_host_cell_index concurrently.
        workers = []
        for _ in range(0, N_THREADS):
            workers.append(threading.Thread(target=closure))

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()

        self.check(results)

    def test_concurrent_pool(self):
        # Defines a thread barrier that will be spawned before parallel execution
        # this increases the probability of concurrent access clashes.
        barrier = threading.Barrier(N_THREADS)

        # This object will be shared by all the threads.
        registry = BoundaryRegistry()

        def test_recipe(
            same_side_active_layer,
            same_side_ghost_layer,
            opposite_side_active_layer,
            opposite_side_ghost_layer,
            weight_same_side_active_layer,
            weight_same_side_ghost_layer,
            weight_opposite_side_active_layer,
            weight_opposite_side_ghost_layer,
            side,
            metadata,
        ): ...

        results = []

        def closure():
            assert "test" not in registry

            # Ensure that all threads reach this point before concurrent execution.
            barrier.wait()
            registry.register("test", test_recipe)
            results.append(registry["test"])

        with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
            futures = [executor.submit(closure) for _ in range(N_THREADS)]

        results = [f.result() for f in futures]
        self.check(results)
