# setup.py

# Copyright (c) 2020, Aarón Villanueva
# License: MIT
#   Full Text: https://gitlab.com/aaronuv/arby/-/blob/master/LICENSE


# ==========================
# Docs
# ==========================


"""Distribute and install Arby."""


# ==========================
# Imports
# ==========================

import os

from setuptools import setup


# ==========================
# Constants
# ==========================


BASE_DIR = os.path.abspath(os.path.dirname(__file__))

arby_init_path = os.path.join(BASE_DIR, "arby", "__init__.py")

REQUIREMENTS = ["numpy", "scipy"]


with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()


with open(arby_init_path, "r") as f:
    for line in f:
        if line.startswith("__version__"):
            _, _, ARBY_VERSION = line.replace('"', "").split()
            break


# ==========================
# Functions
# ==========================


setup(
    name="arby",
    version=ARBY_VERSION,
    description="Build reduced bases and surrogate models in Python",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author="Aarón Villanueva",
    author_email="aaron.villanueva@unc.edu.ar",
    url="https://gitlab.com/aaronuv/arby",
    packages=["arby"],
    install_requires=REQUIREMENTS,
    license="The MIT License",
    keywords=["surrogates", "reduced basis", "empirical interpolation"],
    classifiers=[
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Scientific/Engineering",
    ],
)
