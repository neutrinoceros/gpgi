import numpy
from Cython.Build import cythonize
from distutils.extension import Extension
from setuptools import setup

ext_modules = ["gpgi/clib/_indexing.pyx"]


setup(
    name="Hello world app",
    ext_modules=cythonize(
        Extension(
            "gpgi.clib._indexing",
            ["gpgi/clib/_indexing.pyx"],
            include_dirs=[numpy.get_include()],
        ),
        compiler_directives={"language_level": 3},
    ),
)
