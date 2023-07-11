import os
import sys
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize
from setuptools import setup


def make_ext(path: str) -> Extension:
    name, _ = os.path.splitext(path)
    name = name.replace("/", ".")

    if sys.version_info >= (3, 9):
        # keep in sync with runtime requirements (pyproject.toml)
        define_macros = [("NPY_TARGET_VERSION", "NPY_1_18_API_VERSION")]
    else:
        define_macros = []

    return Extension(
        name,
        sources=[f"src/{path}"],
        include_dirs=[numpy.get_include()],
        define_macros=define_macros,
    )


setup(
    ext_modules=cythonize(
        [
            make_ext("gpgi/clib/_indexing.pyx"),
            make_ext("gpgi/clib/_deposition_methods.pyx"),
        ],
        compiler_directives={"language_level": 3},
    ),
)
