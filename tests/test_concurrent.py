# exercise some thread concurrency
# https://py-free-threading.github.io/debugging/


import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import gpgi

prng = np.random.default_rng()


def test__setup_host_cell_index_concurrent_threading():
    # Defines a thread barrier that will be spawned before parallel execution
    # this increases the probability of concurrent access clashes.
    n_threads = 10
    barrier = threading.Barrier(n_threads)

    # This object will be shared by all the threads.
    ds = gpgi.load(
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

    results = []

    def closure():
        # Ensure that all threads reach this point before concurrent execution.
        barrier.wait()
        hci = ds._setup_host_cell_index()
        assert ds.host_cell_index is hci
        results.append(id(hci))

    # Spawn n threads that call call_unsafe concurrently.
    workers = []
    for _ in range(0, n_threads):
        workers.append(threading.Thread(target=closure))

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()

    # Check results: verify that all threads see the same array
    assert len(set(results)) == 1


def test__setup_host_cell_index_concurrent_pool():
    # Defines a thread barrier that will be spawned before parallel execution
    # this increases the probability of concurrent access clashes.
    n_threads = 10
    barrier = threading.Barrier(n_threads)

    # This object will be shared by all the threads.
    ds = gpgi.load(
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

    results = []

    def closure():
        # Ensure that all threads reach this point before concurrent execution.
        barrier.wait()
        hci = ds._setup_host_cell_index()
        assert ds.host_cell_index is hci
        results.append(id(hci))

    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(closure) for _ in range(n_threads)]

    results = [f.result() for f in futures]

    # Check results: verify that all threads see the same array
    assert len(set(results)) == 1
