# GPGI
A **G**eneric **P**article + **G**rid data **I**nterface

This small Python library implements fundamental grid deposition algorithms to analyse grid + particle datasets, with an emphasize on performance.
Core algorithms are writen as Cython extensions.

### Installation

```shell
python -m pip install git@https://github.com/neutrinoceros/gpgi.git
```



### Usage

The API consists in a `load` function, which returns a `Dataset` object.
Data can be 1D, 2D, or 3D and is validated at load time for consistency,
in particular with the chosen geometry (e.g. cartesian, polar, cylindrical, spherical).

**Load** data

```python
import numpy as np
import gpgi

nx = ny = 64
nparticles = 600_000

prng = np.random.RandomState(0)
ds = gpgi.load(
    geometry="cartesian",
    grid={
        "cell_edges": {
            "x": np.linspace(-1, 1, nx),
            "y": np.linspace(-1, 1, ny),
        },
    },
    particles={
        "coordinates": {
            "x": 2 * (prng.normal(0.5, 0.25, nparticles) % 1 - 0.5),
            "y": 2 * (prng.normal(0.5, 0.25, nparticles) % 1 - 0.5),
        },
        "fields": {
            "mass": np.ones(nparticles),
        },
    },
)
```
The `Dataset` object holds a `grid` and a `particle` attribute,
which both hold a `fields` attribute for accessing their data.
But more importantly, the `Dataset` has a `deposit` method to
translate particle fields to the grid formalism.

**Deposit Particle fields on the grid **

```python
particle_mass = ds.deposit("mass", method="particle_in_cell")  # or "pic" for shorts
```

**Visualize**
In this example we'll use `matplotlib` for rendering, but note that `matplotlib` is not a dependency to `gpgi`
```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set(aspect=1, xlabel="x", ylabel="y")

im = ax.pcolormesh(
    "x",
    "y",
    particle_mass,
    data=ds.grid.cell_edges,
    cmap="viridis",
 )
fig.colorbar(im, ax=ax)
```

<p align="center">
    <img src="https://raw.githubusercontent.com/neutrinoceros/gpgi/main/tests/pytest_mpl_baseline/test_readme_example.png" width="400"></a>
</p>

The example script given here takes about a second (top to bottom).
