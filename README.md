# GPGI
[![PyPI](https://img.shields.io/pypi/v/gpgi.svg?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/gpgi/)
[![PyPI](https://img.shields.io/badge/requires-Python%20≥%203.9-blue?logo=python&logoColor=white)](https://pypi.org/project/gpgi/)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/neutrinoceros/gpgi/main.svg)](https://results.pre-commit.ci/latest/github/neutrinoceros/gpgi/main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A **G**eneric **P**article + **G**rid data **I**nterface

This small Python library implements fundamental grid deposition algorithms to
analyse (rectilinear) grid + particle datasets, with an emphasize on
performance. Core algorithms are implemented as Cython extensions.

## Installation

```shell
python -m pip install --upgrade pip
python -m pip install gpgi
```

## Supported applications

A rectilinear grid is defined as 1D arrays representing cell left edges in each directions. Note that the last point of such an array is interpreted as the right edge of the rightmost cell, so for instance, a 1D grid containing 100 cells is defined by 101 edges.

Particles are defined as points that live on the grid.

Deposition is the action of going from particule description
to a grid description of a field.
It is useful to analyze, compare and combine simulation data that exists in a combination of the two formalisms.
This process is not reversible as it degrades information.

For instance, here's a simple overlay of a particle set (red dots)
against a background that represents the deposited particle count.
<p align="center">
    <img src="https://raw.githubusercontent.com/neutrinoceros/gpgi/main/tests/pytest_mpl_baseline/test_2D_deposit_pic.png" width="600"></a>
</p>

This example illustrates the simplest possible deposition method "Particle in Cell", in which each particle contributes only to the cell
that contains it.

More refined methods are also available.
### Supported deposition methods
| method name             | abreviated name | order |
|-------------------------|:---------------:|:-----:|
| Particle in Cell        | PIC             | 0     |
| Cloud in Cell           | CIC             | 1     |
| Triangular Shaped Cloud | TSC             | 2     |


### Supported boundary conditions
With CIC and TSC deposition, particles contribute to cells neighbouring the one
that contains them. This means that particles that live in the outermost layer of the
domain are partly smoothed out of it.


> 🚧 This section is under construction 🚧
>
> In a future version, I intend to allow special treatments for these lost bits,
> in particular, periodic boundaries.

### Supported geometries
| geometry name | axes order                  |
|---------------|-----------------------------|
| cartesian     | x, y, z                     |
| polar         | radius, z, azimuth          |
| cylindrical   | radius, azimuth, z          |
| spherical     | radius, colatitude, azimuth |
| equatorial    | radius, azimuth, latitude   |

## Time complexity

An important step in perfoming deposition is to associate particle indices to cell indices. This step is called "particle indexing".
In directions where the grid is uniformly stepped (if any), indexing a particle is an O(1) operation.
In the more general case, indexing is performed by bisection, which is a O(log(nx))) operation (where nx represents the number of cells in the direction of interest).


## Usage

The API consists in a `load` function, which returns a `Dataset` object.

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

**Deposit Particle fields on the grid**

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
    particle_mass.T,
    data=ds.grid.cell_edges,
    cmap="viridis",
)
fig.colorbar(im, ax=ax)
```

<p align="center">
    <img src="https://raw.githubusercontent.com/neutrinoceros/gpgi/main/tests/pytest_mpl_baseline/test_readme_example.png" width="600"></a>
</p>

The example script given here takes about a second (top to bottom).
