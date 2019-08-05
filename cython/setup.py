from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

setup(
    ext_modules = cythonize(Extension("matrices",sources=["matrices.pyx"],extra_link_args=["-O3","-fopenmp"],language="c++",extra_compile_args=["-O3", "-march=native","-fopenmp"]))
    
)