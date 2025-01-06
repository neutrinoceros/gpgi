import os
import re
import sys
import tomllib
from dataclasses import dataclass
from distutils.extension import Extension
from enum import IntEnum
from pathlib import Path
from typing import Literal

import numpy as np
from Cython.Build import cythonize
from setuptools import setup
from wheel.bdist_wheel import bdist_wheel

PROJECT_DIR = Path(__file__).parent
SRC_DIR = PROJECT_DIR / "src" / "gpgi"

PYPROJECT_TOML = PROJECT_DIR / "pyproject.toml"


class Level(IntEnum):
    MAJOR = 0x01000000
    MINOR = 0x00010000
    MICRO = 0x00000100
    ALPHA = 0xA0
    BETA = 0xB0
    CANDIDATE = 0xC0
    FINAL = 0xF0


@dataclass(frozen=True, slots=True)
class PythonVersion:
    major: int
    minor: int
    micro: int = 0
    release_level: Literal[Level.ALPHA, Level.BETA, Level.CANDIDATE, Level.FINAL] = (
        Level.FINAL
    )
    serial: int = 0

    def to_hex(self) -> str:
        hex_ = hex(
            self.major * Level.MAJOR
            + self.minor * Level.MINOR
            + self.micro * Level.MICRO
            + self.release_level
            + self.serial
        )
        MIN_SIZE = 8
        if len(hex_.removeprefix("0x")) < MIN_SIZE:
            return hex_[:2] + hex_[2:].zfill(MIN_SIZE)
        else:
            return hex_

    def to_cp_tag(self) -> str:
        return f"cp{self.major}{self.minor}"


with open(PYPROJECT_TOML, "rb") as fh:
    conf = tomllib.load(fh)
if (
    match := re.match(r"^>=\w*(?P<version>3\.\d+)$", conf["project"]["requires-python"])
) is None:
    raise RuntimeError
minimal_python_version_str = match.group("version")
major, minor = (int(_) for _ in minimal_python_version_str.split("."))
minimal_python_version = PythonVersion(major, minor)


class bdist_wheel_abi3(bdist_wheel):
    def get_tag(self):
        python, abi, plat = super().get_tag()

        if python.startswith("cp"):
            return minimal_python_version.to_cp_tag(), "abi3", plat

        return python, abi, plat


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
                    ("Py_LIMITED_API", minimal_python_version.to_hex()),
                ],
                py_limited_api=True,
            )
        ],
        compiler_directives={"language_level": 3},
    )

setup(ext_modules=ext_modules, cmdclass={"bdist_wheel": bdist_wheel_abi3})
