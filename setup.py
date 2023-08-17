from setuptools import setup, Extension, find_packages
import numpy as np

PACKAGE_NAME = "platekinematics"

setup(
    name=PACKAGE_NAME,
    version="0.1",
    description="Tools for easy handling of plate kinematic functions",
    author="Valentina Espinoza",
    author_email="valentina0694@hotmail.com",
    include_dirs=[np.get_include(), "src/vcpkg/installed/x64-windows/include"],
    package_dir={'platekinematics': 'src',
                 'maths': 'src',
                 'spherical_functions': 'src'},
    ext_package="platekinematics",
    ext_modules=[Extension("maths", [r"src\maths.c"]),
                 Extension("spherical_functions", [r"src\spherical_functions.c"]),
                 Extension("build_ensemble", 
                           sources = [r"src\build_ensemble.c"], 
                           define_macros=[('GSL_DLL', None)],
                           library_dirs=['src/vcpkg/installed/x64-windows/lib'],
                           libraries=['gsl', 'gslcblas'], 
                           )
                 ]
)