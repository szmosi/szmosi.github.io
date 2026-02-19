from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import platform

# Determine compiler flags based on Operating System
if platform.system() == "Windows":
    # /O2: Maximum speed optimization
    # /arch:AVX2: Enables 256-bit SIMD instructions (crucial for MatMul/Sum)
    # /fp:fast: Speeds up floating-point math by ignoring strict IEEE standards
    compile_flags = ['/openmp', '/O2', '/arch:AVX2', '/fp:fast']
    link_flags = []
else:
    # -O3: Aggressive optimization level
    # -march=native: Optimizes specifically for YOUR computer's CPU features
    # -ffast-math: Speeds up floating-point math
    compile_flags = ['-fopenmp', '-O3', '-march=native', '-ffast-math']
    link_flags = ['-fopenmp']

ext_modules = [
    Extension(
        "fast_code",
        ["fast_code.pyx"],
        extra_compile_args=compile_flags,
        extra_link_args=link_flags,
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name="FastCode",
    ext_modules=cythonize(ext_modules, compiler_directives={'language_level': "3"}),
)