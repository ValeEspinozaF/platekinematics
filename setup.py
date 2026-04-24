from pathlib import Path
import os

import numpy as np
from setuptools import Extension, find_packages, setup


PACKAGE_NAME = "platekinematics"


def _gsl_library_dirs():
    # Allow CI/build systems to inject the vcpkg lib directory.
    candidates = [
        os.environ.get("PK_GSL_LIB_DIR"),
        str(Path("src") / "vcpkg" / "installed" / "x64-windows" / "lib"),
        "lib",
    ]
    return [path for path in candidates if path and Path(path).is_dir()]


setup(
    name=PACKAGE_NAME,
    version="0.1",
    description="Tools for easy handling of plate kinematic functions",
    author="Valentina Espinoza",
    author_email="valentina0694@hotmail.com",
    packages=find_packages(),
    include_package_data=True,
    python_requires=">=3.8",
    ext_package="platekinematics",
    ext_modules=[Extension("spherical_functions", [r"src/spherical_functions.c"], include_dirs=[np.get_include(), r"include"],),
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
                           library_dirs=_gsl_library_dirs(),
                           libraries=['gsl', 'gslcblas'],
                           include_dirs=[np.get_include(), r"include"],
                           ),
                 ],
)