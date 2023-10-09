#define _USE_MATH_DEFINES
#include <Python.h>
#include <math.h>
//#define M_PI 3.1415926535

//#include "activate-overloads.h"

/* Main functions */
inline double to_degrees(double radians) {
    return radians * (180.0 / M_PI);
}


inline double to_radians(double degrees) {
    return degrees * (M_PI / 180.0);
}


double * sph2cart(double lon, double lat, double mag) {
    static double r[3];
    double lon_rad = to_radians(lon);
    double lat_rad = to_radians(lat);

    r[0] = mag * cos(lon_rad) * cos(lat_rad);
    r[1] = mag * sin(lon_rad) * cos(lat_rad);
    r[2] = mag * sin(lat_rad);

    return r;
}


double * cart2sph(double x, double y, double z) {
    static double r[3];
    r[0] = to_degrees(atan2(y, x)); //lon
    r[1] = to_degrees(atan2(z, sqrt(x * x + y * y))); //lat
    r[2] = sqrt(x * x + y * y + z * z); //angle
    return r;
}


static PyObject *py_to_degrees(PyObject *self, PyObject *args) {

  /* Declare variables */
  double n_num, result;

  /* Parse argument from python to local variable (n_num) */
  if (!PyArg_ParseTuple(args, "f", &n_num)) {
    return NULL;
  }

  /* Assign value to output variable */
  result = to_degrees(n_num);

  /* Return */
  return Py_BuildValue("f", result);
}


static PyObject *py_to_radians(PyObject *self, PyObject *args) {
  double n_num, result;

  if (!PyArg_ParseTuple(args, "f", &n_num)) {
    return NULL;
  }

  result = to_radians(n_num);
  return Py_BuildValue("f", result);
}


static PyObject *py_sph2cart(PyObject *self, PyObject *args) {
  double lon, lat, mag;
  double *result;

  if (!PyArg_ParseTuple(args, "fff", &lon, &lat, &mag)) {
    return NULL;
  }

  result = sph2cart(lon, lat, mag);
  return Py_BuildValue("fff", result[0], result[1], result[2]);
}

static PyObject *py_cart2sph(PyObject *self, PyObject *args) {
  double x, y, z;
  double *result;

  if (!PyArg_ParseTuple(args, "fff", &x, &y, &z)) {
    return NULL;
  }

  result = cart2sph(x, y, z);
  return Py_BuildValue("fff", result[0], result[1], result[2]);
}


/* Methods contained in the module */
static PyMethodDef sphericalMethods[] = {
  {"to_degrees", py_to_degrees, METH_VARARGS, "Convert angles from radians to degrees."},
  {"to_radians", py_to_radians, METH_VARARGS, "Convert angles from degrees to radians."},
  {"sph2cart", py_sph2cart, METH_VARARGS, "Convert spherical coordinates to cartesian coordinates. Both input and outputs are expressed in degrees. "},
  {"cart2sph", py_cart2sph, METH_VARARGS, "Convert cartesian coordinates to spherical coordinates. Both input and outputs are expressed in degrees. "},
  {NULL, NULL, 0, NULL}
};


/* Module definition */
static struct PyModuleDef spherical_functions = {
  PyModuleDef_HEAD_INIT,
  "spherical_functions",
  "Functions for coordinate system transformations in spherical geometry.",
  -1,
  sphericalMethods
};

PyMODINIT_FUNC PyInit_spherical_functions(void)
{
    return PyModule_Create(&spherical_functions);
}