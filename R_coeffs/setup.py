from setuptools import setup, Extension
from Cython.Build import cythonize

setup(name="rcoeffs_cy", ext_modules=cythonize("rcoeffs_cy.pyx", language_level=3))
