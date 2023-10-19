from setuptools import setup, Extension, find_packages
import numpy as np

PACKAGE_NAME = "platekinematics"

build_ = Extension("build_ensemble", [r"src\build_ensemble.c"], 
                           #define_macros=[('GSL_DLL', None)],
                           library_dirs=[r'src/vcpkg/installed/x64-windows/lib'],
                           libraries=['gsl', 'gslcblas']
                           )

build_2 = Extension("methods", [r"src/types/covariance.c",
                                r"src/types/finrot.c",
                                r"src/types/eulervec.c",
                                r"src/type_methods/covariance_methods.c",
                                r"src/type_methods/eulervec_methods.c",
                                r"src/type_methods/finrot_methods.c",
                                r"src/spherical_functions.c", 
                                r"src/parse_array.c",
                                r"src/build_ensemble.c",
                                r"src/type_conversions/covariance_conversions.c",
                                r"src/type_conversions/finrot_conversions.c",
                                r"src/average_vector.c", 
                                r"src/average_eulervec.c", 
                                r"src/average_finrot.c", 
                                r"src/average_ensemble.c", 
                                r"src/pk_structs.c",
                                ],
                           library_dirs=[r'src/vcpkg/installed/x64-windows/lib'],
                           libraries=['gsl', 'gslcblas']
                           )

build_4 = Extension("maths", [r"src/maths.c"],
                           library_dirs=[r'src/vcpkg/installed/x64-windows/lib'],
                           libraries=['gsl', 'gslcblas']
                           )
                 
                 

setup(
    name=PACKAGE_NAME,
    version="0.1",
    description="Tools for easy handling of plate kinematic functions",
    author="Valentina Espinoza",
    author_email="valentina0694@hotmail.com",
    include_dirs=[np.get_include(), r"src/vcpkg/installed/x64-windows/include"],
    package_dir={'platekinematics': 'src'},
    ext_package="platekinematics",
    ext_modules=[Extension("spherical_functions", [r"src/spherical_functions.c"]),
                 Extension("pk_structs", [r"src/types/covariance.c",
                                          r"src/types/finrot.c",
                                          r"src/types/eulervec.c",
                                          r"src/type_conversions/covariance_conversions.c",
                                          r"src/type_conversions/finrot_conversions.c",
                                          r"src/parse_array.c",
                                          r"src/build_ensemble.c",
                                          r"src/type_methods/covariance_methods.c",
                                          r"src/type_methods/eulervec_methods.c",
                                          r"src/type_methods/finrot_methods.c",
                                          r"src/spherical_functions.c",
                                          r"src/pk_structs.c",
                                          r"src/average_vector.c", 
                                          r"src/average_eulervec.c", 
                                          r"src/average_finrot.c", 
                                          ],
                           library_dirs=[r'src/vcpkg/installed/x64-windows/lib'],
                           libraries=['gsl', 'gslcblas'],
                           ),
                 ]
)