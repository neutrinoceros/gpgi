project('gpgi', 'c', 'cython', meson_version: '>=1.5.0')

py = import('python').find_installation(pure: false)

min_numpy_version = '1.25.0'  # keep in sync with pyproject.toml
cy = meson.get_compiler('cython')
cython = find_program(cy.cmd_array()[0])


# https://github.com/mesonbuild/meson/issues/9598
incdir_numpy = run_command(py,
  [
    '-c',
    'import os; os.chdir(".."); import numpy; print(numpy.get_include())'
  ],
  check: true
).stdout().strip()

inc_np = include_directories(incdir_numpy)

py.extension_module(
    'gpgi._lib',
    'src/gpgi/_lib.pyx',
    install: true,
    include_directories : [inc_np],
)
