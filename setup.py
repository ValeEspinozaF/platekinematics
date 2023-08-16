from setuptools import setup, Extension, find_packages

PACKAGE_NAME = "platekinematics"

setup(
    name=PACKAGE_NAME,
    version="0.1",
    description="Tools for easy handling of plate kinematic functions",
    author="Valentina Espinoza",
    author_email="valentina0694@hotmail.com",
    package_dir={'platekinematics': 'src',
                 'maths': 'src',
                 'spherical_functions': 'src'},
    ext_package="platekinematics",
    ext_modules=[Extension("maths", ["src\maths.c"]),
                 Extension("spherical_functions", ["src\spherical_functions.c"]),
                 ]
)