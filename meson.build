project(
  'gpgi',
  'c',
  'cython',
  # https://mesonbuild.com/meson-python/reference/meson-compatibility.html#cmdoption-arg-1.6.0
  meson_version: '>=1.6.0',
)

# detect missing dependencies early
py = import('python').find_installation(pure: false)
cy = meson.get_compiler('cython')
cython = find_program(cy.cmd_array()[0])

npy_c_args = [
  # keep in sync with pyproject.toml
  '-DNPY_TARGET_VERSION=NPY_1_25_API_VERSION',
  '-DNPY_NO_DEPRECATED_API=NPY_1_25_API_VERSION',
]

npy_dep = declare_dependency(
  dependencies: dependency('numpy'),
  compile_args: npy_c_args,
)

if get_option('pylib')
  py.install_sources('src/gpgi/_lib.py', subdir: 'gpgi')
else
  py.extension_module(
      '_lib',
      'src/gpgi/_lib.pyx',
      subdir: 'gpgi',
      install: true,
      dependencies : npy_dep,
      c_args: npy_c_args,
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
