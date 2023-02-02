from distutils.extension import Extension

import numpy
from Cython.Build import cythonize
from setuptools import setup

ext_modules = ["gpgi/clib/_indexing.pyx"]


setup(
    ext_modules=cythonize(
        [
            Extension(
                "gpgi.clib._indexing",
                ["gpgi/clib/_indexing.pyx"],
                include_dirs=[numpy.get_include()],
            ),
            Extension(
                "gpgi.clib._deposition_methods",
                ["gpgi/clib/_deposition_methods.pyx"],
                include_dirs=[numpy.get_include()],
            ),
        ],
        compiler_directives={"language_level": 3},
        annotate=True,
    ),
)
