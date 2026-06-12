# exercise some thread concurrency
# https://py-free-threading.github.io/debugging/


import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import numpy.testing as npt
import pytest

import gpgi
from gpgi import Dataset
from gpgi._boundaries import BoundaryRegistry
from gpgi._typing import D2, LockArg, f64

prng = np.random.default_rng()


N_THREADS = 10


def random_dataset() -> Dataset[D2, f64]:
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
            },
            "fields": {
                "mass": prng.random(10_000),
            },
        },
    )


class TestSetupHostCellIndex:
    def check(self, results: Sequence[int]) -> None:
        # Check results: verify that all threads see the same array
        assert len(set(results)) == 1
        # check that we indeed collected ids, not None
        assert isinstance(results[0], int)

    def test_concurrent_threading(self) -> None:
        # Defines a thread barrier that will be spawned before parallel execution
        # this increases the probability of concurrent access clashes.
        barrier = threading.Barrier(N_THREADS)

        # This object will be shared by all the threads.
        ds = random_dataset()

        results: list[int] = []

        def closure() -> None:
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

    def test_concurrent_pool(self) -> None:
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


@pytest.mark.parametrize("lock", ["per-instance", None, threading.Lock()])
class TestDeposit:
    def check(self, results: Sequence[np.ndarray[D2, np.dtype[f64]]]) -> None:
        ref = results[0]
        for res in results[1:]:
            npt.assert_array_equal(res, ref)
        assert len({id(res) for res in results}) == len(results)

    def test_concurrent_threading(self, lock: LockArg) -> None:
        # Defines a thread barrier that will be spawned before parallel execution
        # this increases the probability of concurrent access clashes.
        barrier = threading.Barrier(N_THREADS)

        # This object will be shared by all the threads.
        ds = random_dataset()

        results: list[np.ndarray[D2, np.dtype[f64]]] = []

        def closure() -> None:
            # Ensure that all threads reach this point before concurrent execution.
            barrier.wait()
            dep = ds.deposit("mass", method="nearest_grid_point", lock=lock)
            results.append(dep)

        # Spawn n threads that call _setup_host_cell_index concurrently.
        workers = []
        for _ in range(0, N_THREADS):
            workers.append(threading.Thread(target=closure))

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()

        self.check(results)

    def test_concurrent_pool(self, lock: LockArg) -> None:
        # Defines a thread barrier that will be spawned before parallel execution
        # this increases the probability of concurrent access clashes.
        barrier = threading.Barrier(N_THREADS)

        # This object will be shared by all the threads.
        ds = random_dataset()

        def closure():
            # Ensure that all threads reach this point before concurrent execution.
            barrier.wait()
            return ds.deposit("mass", method="nearest_grid_point", lock=lock)

        with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
            futures = [executor.submit(closure) for _ in range(N_THREADS)]

        results = [f.result() for f in futures]
        self.check(results)


class TestBoundaryRegistry:
    def check(self, results: Sequence[str]) -> None:
        # only one thread can succeed registration, all others should raise.
        expected_msg = (
            "Another function is already registered with key='test'. "
            "If you meant to override the existing function, "
            "consider setting allow_unsafe_override=True"
        )
        assert len(results) == N_THREADS - 1
        assert results.count(expected_msg) == N_THREADS - 1

    def test_concurrent_threading(self) -> None:
        # Defines a thread barrier that will be spawned before parallel execution
        # this increases the probability of concurrent access clashes.
        barrier = threading.Barrier(N_THREADS)

        # This object will be shared by all the threads.
        registry = BoundaryRegistry()

        results: list[str] = []

        def closure():
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

            assert "test" not in registry

            # Ensure that all threads reach this point before concurrent execution.
            barrier.wait()
            try:
                registry.register("test", test_recipe)
            except ValueError as exc:
                msg, *_ = exc.args
                results.append(msg)
            assert "test" in registry

        # Spawn n threads that call _setup_host_cell_index concurrently.
        workers = []
        for _ in range(0, N_THREADS):
            workers.append(threading.Thread(target=closure))

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()

        self.check(results)

    def test_concurrent_pool(self) -> None:
        # Defines a thread barrier that will be spawned before parallel execution
        # this increases the probability of concurrent access clashes.
        barrier = threading.Barrier(N_THREADS)

        # This object will be shared by all the threads.
        registry = BoundaryRegistry()

        def closure():
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

            assert "test" not in registry

            # Ensure that all threads reach this point before concurrent execution.
            barrier.wait()
            registry.register("test", test_recipe)

        with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
            futures = [executor.submit(closure) for _ in range(N_THREADS)]

        assert "test" in registry
        exceptions = [f.exception() for f in futures]
        results = [exc.args[0] for exc in exceptions if exc is not None]
        self.check(results)
