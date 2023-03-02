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
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            ),
            Extension(
                "gpgi.clib._deposition_methods",
                ["gpgi/clib/_deposition_methods.pyx"],
                include_dirs=[numpy.get_include()],
                define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            ),
        ],
        # annotate=True, # uncomment to produce html reports
    ),
)
