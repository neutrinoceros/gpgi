import os
import sys
from distutils.extension import Extension
from pathlib import Path

import numpy as np
from Cython.Build import cythonize
from setuptools import setup

SRC_DIR = Path(__file__).parent / "src" / "gpgi"

if os.getenv("GPGI_PY_LIB", "0").lower() in ("1", "true"):
    if not SRC_DIR.joinpath("_lib.py").exists():
        raise RuntimeError(
            "GPGI's pure Python implementation can only be built "
            "from the development version in editable mode."
        )
    ext_modules = []
    pattern = "*.pyd" if sys.platform.startswith("win") else "*.so"
    for sofile in SRC_DIR.glob(pattern):
        os.remove(sofile)
else:
    ext_modules = cythonize(
        [
            Extension(
                "gpgi._lib",
                sources=["src/gpgi/_lib.pyx"],
                include_dirs=[np.get_include()],
                define_macros=[
                    # keep in sync with runtime requirements (pyproject.toml)
                    ("NPY_TARGET_VERSION", "NPY_1_25_API_VERSION"),
                    ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
                ],
            )
        ],
        compiler_directives={"language_level": 3},
    )

setup(ext_modules=ext_modules)
