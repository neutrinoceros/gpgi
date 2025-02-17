project('gpgi', 'c', 'cython', meson_version: '>=1.5.0')

# detect missing dependencies early
py = import('python').find_installation(pure: false)
cy = meson.get_compiler('cython')
cython = find_program(cy.cmd_array()[0])

numpy_compile_flags = [
  # keep in sync with pyproject.toml
  '-DNPY_TARGET_VERSION=NPY_1_25_API_VERSION',
  '-DNPY_NO_DEPRECATED_API=NPY_1_25_API_VERSION',
]

# copied from PyWavelets
# https://github.com/PyWavelets/pywt/blob/becef5486174c77727ea2a60e6392744b2cc1d4a/pywt/_extensions/meson.build#L13
# When building against numpy 1.x is dropped, this can be simplified: the else branch
# becomes unreachable, so `required: false` can be omitted.

_numpy_dep = dependency('numpy', required: false)
if _numpy_dep.found()
  np_dep = declare_dependency(dependencies: _numpy_dep, compile_args: numpy_compile_flags)
else
  # For cross-compilation it is often not possible to run the Python interpreter
  # in order to retrieve numpy's include directory. It can be specified in the
  # cross file instead:
  #   [properties]
  #   numpy-include-dir = /abspath/to/host-pythons/site-packages/numpy/core/include
  #
  # This uses the path as is, and avoids running the interpreter.
  incdir_numpy = meson.get_external_property('numpy-include-dir', 'not-given')
  if incdir_numpy == 'not-given'
    incdir_numpy = run_command(py,
      [
        '-c',
        '''import os
import numpy as np
try:
  incdir = os.path.relpath(np.get_include())
except Exception:
  incdir = np.get_include()
print(incdir)
    '''
      ],
      check: true
    ).stdout().strip()
  endif
  inc_np = include_directories(incdir_numpy)
  np_dep = declare_dependency(include_directories: inc_np, compile_args: numpy_compile_flags)
endif

if get_option('pylib')
  py.install_sources('src/gpgi/_lib.py', subdir: 'gpgi')
else
  py.extension_module(
      '_lib',
      'src/gpgi/_lib.pyx',
      subdir: 'gpgi',
      install: true,
      dependencies : [np_dep],
      c_args: numpy_compile_flags,
      limited_api: '3.11', # keep in sync with requires-python (pyproject.toml)
  )
endif

# meson doesn't have an equivalent to setuptools.packages.find
# at the time of writing, so, listing every module explicitly here
py.install_sources(
    'src/gpgi/__init__.py',
    'src/gpgi/_boundaries.py',
    'src/gpgi/_data_types.py',
    'src/gpgi/_lib.pyi',
    'src/gpgi/_load.py',
    'src/gpgi/_spatial_data.py',
    'src/gpgi/_typing.py',
    'src/gpgi/py.typed',
    'src/gpgi/typing.py',
    subdir: 'gpgi'
)
