# GPGI
[![PyPI](https://img.shields.io/pypi/v/gpgi.svg?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/gpgi/)
[![PyPI](https://img.shields.io/badge/requires-Python%20≥%203.8-blue?logo=python&logoColor=white)](https://pypi.org/project/gpgi/)
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
    <img src="https://raw.githubusercontent.com/neutrinoceros/gpgi/main/tests/pytest_mpl_baseline/test_2D_deposit_ngp.png" width="600"></a>
</p>

This example illustrates the simplest possible deposition method "Particle in Cell", in which each particle contributes only to the cell
that contains it.

More refined methods are also available.
### Supported deposition methods
| method name             | abreviated name | order |
|-------------------------|:---------------:|:-----:|
| Nearest Grid Point      | NGP             | 0     |
| Cloud in Cell           | CIC             | 1     |
| Triangular Shaped Cloud | TSC             | 2     |

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
The `Dataset` object holds a `grid` and a `particles` attribute,
which both hold a `fields` attribute for accessing their data.
But more importantly, the `Dataset` has a `deposit` method to
translate particle fields to the grid formalism.

**Deposit Particle fields on the grid**

```python
particle_mass = ds.deposit("mass", method="nearest_grid_point")  # or "ngp" for shorts
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


### Supplying arbitrary metadata
*new in gpgi 0.4.0*

`Dataset` objects have a special attribute `metadata` which is a dictionary with string keys.
This attribute is meant to hold any special metadata that may be relevant for labelling or processing (e.g. simulation time, author, ...).
Metadata can be supplied at load time as
```python
ds = gpgi.load(
    geometry="cartesian",
    grid=...,
    particles=...,
    metadata={"simulation_time": 12.5, "author": "Clément Robert"}
)
```


### Boundary conditions
*new in gpgi 0.5.0*

With CIC and TSC deposition, particles contribute to cells neighbouring the one
that contains them. For particles that live in the outermost layer of the
domain, this means some of their contribution is lost. This behaviour
corresponds to the default `'open'` boundary condition, but `gpgi` has builtin
support for more conservative boundary conditions.

Boundary conditions can selected per field, per axis and per side. Builtin
recipes all perform linear combinations of ghost layers (same-side and opposite
side) and active domain layers (same-side and opposite side), and replace the
same-side active layer with the result.

User-selected boundary conditions take the form of an optional argument to
`Dataset.deposit`, as dictionnary with keys being axes names, and values being
2-tuples of boundary conditions names (for left and right side respectively).
For instance, here's how one would require periodic boundary conditions on all axes:

```python
ds.deposit(
    "mass",
    method="cic",
    boundaries={
        "x": ("periodic", "periodic"),
        "y": ("periodic", "periodic"),
    }
)
```
Unspecified axes will use the default `'open'` boundary.


#### Builtin recipes

| boundary conditions     | description                                            | conservative ? |
|-------------------------|--------------------------------------------------------|:--------------:|
| open (default)          | no special treatment                                   | no             |
| periodic                | add opposite ghost layer to the active domain          | yes            |
| wall                    | add same-side ghost layer to the active domain         | yes            |
| antisymmetric           | substract same-side ghost layer from the active domain | no             |

#### Define custom recipes

`gpgi`'s boundary recipes can be customized. Let's illustrate this feature with a simple example.
Say we want to fix the value of the deposited field in some outer layer.
This is done by defining a new function on the user side:

```python
def ones(
    same_side_active_layer,
    same_side_ghost_layer,
    opposite_side_active_layer,
    opposite_side_ghost_layer,
    side,
    metadata,
):
   return 1.0
```
where all first four arguments are `numpy.ndarray` objects with the same shape,
to which the return value must be broadcastable, `side` can only be either
`"left"` or `"right"`, and `metadata` is the special `Dataset.metadata`
attribute. Not all arguments need be used in the body of the function, but this
signature is required.

The method must then be registered as a boundary condition recipe as
```python
ds.boundary_recipes.register("ones", ones)
```
where the associated key (here `"ones"`) is arbitrary. The recipe can now be
used exactly as builtin ones, and all of them can be mixed arbitrarily.
```python
ds.deposit(
    "mass",
    method="cic",
    boundaries={
        "x": ("ones", "wall"),
        "y": ("periodic", "periodic"),
    }
)
```
