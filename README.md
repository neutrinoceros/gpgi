# GPGI
[![PyPI](https://img.shields.io/pypi/v/gpgi.svg?logo=pypi&logoColor=white&label=PyPI)](https://pypi.org/project/gpgi/)
[![PyPI](https://img.shields.io/badge/requires-Python%20≥%203.8-blue?logo=python&logoColor=white)](https://pypi.org/project/gpgi/)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/neutrinoceros/gpgi/main.svg)](https://results.pre-commit.ci/latest/github/neutrinoceros/gpgi/main)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/charliermarsh/ruff/main/assets/badge/v1.json)](https://github.com/charliermarsh/ruff)

A **G**eneric **P**article + **G**rid data **I**nterface

This small Python library implements fundamental grid deposition algorithms to
analyse (rectilinear) grid + particle datasets, with an emphasize on
performance. Core algorithms are implemented as Cython extensions.

## Table of Contents

<!-- toc -->

- [Installation](#installation)
- [Supported applications](#supported-applications)
  * [Supported deposition methods](#supported-deposition-methods)
  * [Supported geometries](#supported-geometries)
- [Time complexity](#time-complexity)
- [Usage](#usage)
  * [Supplying arbitrary metadata](#supplying-arbitrary-metadata)
  * [Boundary conditions](#boundary-conditions)
    + [Builtin recipes](#builtin-recipes)
    + [Define custom recipes](#define-custom-recipes)
  * [Weight fields (Depositing intensive quantities)](#weight-fields-depositing-intensive-quantities)
- [Deposition algorithm](#deposition-algorithm)

<!-- tocstop -->

## Installation

```shell
python -m pip install --upgrade pip
python -m pip install gpgi
```

## Supported applications

A rectilinear grid is defined as 1D arrays representing cell left edges in each directions. Note that the last point of such an array is interpreted as the right edge of the rightmost cell, so for instance, a 1D grid containing 100 cells is defined by 101 edges.

Particles are defined as points that live within the grid's bounds.

Deposition is the action of going from particle description
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
    weight_same_side_active_layer,
    weight_same_side_ghost_layer,
    weight_opposite_side_active_layer,
    weight_opposite_side_ghost_layer,
    side,
    metadata,
):
   return 1.0
```
where all first eight arguments are `numpy.ndarray` objects with the same shape (which includes ghost padding !),
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

Note that all first eight arguments in a boundary recipe function should
represent an *extensive* physical quantity (as opposed to *intensive*). When
depositing an *intensive* quantity `u`, a weight field `w` should be supplied
(see next section), in which case, the first four arguments represent `u*w` and
the following four represent *w*, so that *u* can still be obtained within the
function as a ratio if needed.






### Weight fields (Depositing intensive quantities)
*new in gpgi 0.7.0*

Fundamentally, deposition algorithms construct on-grid fields by performing
*summations*. An implication is that the physical quantities being deposited are
required to be *extensive* (like mass or momentum). *Intensive* quantities (like
velocity or temperature) require additional operations, and necessitate the use
of an additional *weight field*.

This section provides showcases their *usage*. For a detailled explanation of the deposition algorithm for intensive quantities, see [Deposition algorithm](#deposition-algorithm).

In order to deposit an *intensive* field (e.g., `vx`), an additional `weight_field` argument must be provided as
```python
ds.deposit(
    "vx",
    method="cic",
    boundaries={
        "y": ("periodic", "periodic"),
        "x": ("antisymmetric", "antisymmetric"),
    },
    weight_field="mass",
    weight_field_boundaries={
        "y": ("periodic", "periodic"),
        "x": ("open", "open"),
    },
)
```

Boundary recipes may be also associated to the weight field with the
`weight_field_boundaries` argument. This arguments becomes *required* if
`boundaries` and `weight_field` are both provided.

Call `help(ds.deposit)` for more detail.



## Deposition algorithm

This section provides details on the general deposition algorithm, as
implemented in `gpgi`.

Without loss of generality, we will illustrate how  an *intensive* field (`v`)
is deposited, since this case requires the most computational steps. As it
happens, depositing an *extensive* field (`w`) separately is actually part of
the algorithm.

**Definitions**

- `v` is an intensive field that we want to deposit on the grid
- `w` is an extensive field that will be used as weights
- `u = v * w` is an extensive equivalent to `v` (conceptually, if `v` is a velocity and `w` is a mass, `u` corresponds to a momentum)

`u(i)`, `v(i)` and `w(i)` are defined for each particle `i`.

We note `U(x)`, `V(x)` and `W(x)` the corresponding on-grid fields, where `V(x)`
is the final output of the algorithm. These are defined at grid cell centers
`x`, within the *active domain*.

Last, we note `U'(x)`, `V'(x)` and `W'(x)` the *raw* deposited fields, meaning
no special treatment is applied to the outermost layers (boundary conditions).
These are defined at grid cell centers, *including one ghost layer* that will be
used to apply boundary conditions.

**Algorithm**

1) `W'` and `U'` are computed as
```
W'(x) = Σ c(i,x) w(i)
U'(x) = Σ c(i,x) w(i) v(i)
```
where `c(i,x)` are geometric coefficients associated with the deposition method. Taking the nearest grid point (NGP) method for illustration, `c(i,x) = 1` if particle `i` is contained in the cell whose center is `x`, and `c(i,x) = 0` elsewhere.

2) boundary conditions are applied
```
W(x) = W_BCO(W', 1, metadata)
U(x) = U_BCO(U', W', metadata)
```
where `W_BCO` and `U_BCO` denote arbitrary boundary condition operators
associated with `W` and `U` respectively, and which take 3 arguments,
representing the field to be transformed, its associated weight field and a
wildcard `metadata` argument which may contain any additional data relevant to
the operator.

Note `1` is used a placeholder "weight" for `W`, for symmetry reasons: all boundary condition operators must expose a similar interface, as explained in [Define custom recipes](#define-custom-recipes).

3) Finally, `V(x)` is obtained as
```
V(x) = (U/W)(x)
```
