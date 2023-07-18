import os
from distutils.extension import Extension

import numpy
from Cython.Build import cythonize
from setuptools import setup


def make_ext(path: str) -> Extension:
    name, _ = os.path.splitext(path)
    name = name.replace("/", ".")

    return Extension(
        name,
        sources=[f"src/{path}"],
        include_dirs=[numpy.get_include()],
        define_macros=[
            # keep in sync with runtime requirements (pyproject.toml)
            ("NPY_TARGET_VERSION", "NPY_1_21_API_VERSION"),
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
        ],
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
