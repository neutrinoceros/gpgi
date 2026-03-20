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

_numpy_dep = dependency('numpy')
np_dep = declare_dependency(dependencies: _numpy_dep, compile_args: numpy_compile_flags)

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
      limited_api: py.language_version(),
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
