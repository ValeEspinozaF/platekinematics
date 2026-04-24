#include "spherical_functions.h"

/* Main functions */
inline double to_degrees(double radians) {
    return radians * (180.0 / M_PI);
}


inline double to_radians(double degrees) {
    return degrees * (M_PI / 180.0);
}


void sph2cart(double lon, double lat, double mag, double out[3]) {
    double lon_rad = to_radians(lon);
    double lat_rad = to_radians(lat);

  out[0] = mag * cos(lon_rad) * cos(lat_rad);
  out[1] = mag * sin(lon_rad) * cos(lat_rad);
  out[2] = mag * sin(lat_rad);
}


void cart2sph(double x, double y, double z, double out[3]) {
  out[0] = to_degrees(atan2(y, x)); //lon
  out[1] = to_degrees(atan2(z, sqrt(x * x + y * y))); //lat
  out[2] = sqrt(x * x + y * y + z * z); //angle
}


static PyObject *py_to_degrees(PyObject *self, PyObject *args) {

  /* Declare variables */
  double n_num, result;

  /* Parse argument from python to local variable (n_num) */
  if (!PyArg_ParseTuple(args, "d", &n_num)) {
    return NULL;
  }

  /* Assign value to output variable */
  result = to_degrees(n_num);

  /* Return */
  return Py_BuildValue("d", result);
}


static PyObject *py_to_radians(PyObject *self, PyObject *args) {
  double n_num, result;

  if (!PyArg_ParseTuple(args, "d", &n_num)) {
    return NULL;
  }

  result = to_radians(n_num);
  return Py_BuildValue("d", result);
}


static PyObject *py_sph2cart(PyObject *self, PyObject *args) {
  double lon, lat, mag;
  double result[3];

  if (!PyArg_ParseTuple(args, "ddd", &lon, &lat, &mag)) {
    return NULL;
  }

  sph2cart(lon, lat, mag, result);
  return Py_BuildValue("ddd", result[0], result[1], result[2]);
}

static PyObject *py_cart2sph(PyObject *self, PyObject *args) {
  double x, y, z;
  double result[3];

  if (!PyArg_ParseTuple(args, "ddd", &x, &y, &z)) {
    return NULL;
  }

  cart2sph(x, y, z, result);
  return Py_BuildValue("ddd", result[0], result[1], result[2]);
}


/* Methods contained in the module */
static PyMethodDef sphericalMethods[] = {
  {"to_degrees", py_to_degrees, METH_VARARGS, "to_degrees(angle_radians) -> float\n\nConvert an angle from radians to degrees."},
  {"to_radians", py_to_radians, METH_VARARGS, "to_radians(angle_degrees) -> float\n\nConvert an angle from degrees to radians."},
  {"sph2cart", py_sph2cart, METH_VARARGS, "sph2cart(lon, lat, magnitude) -> tuple[float, float, float]\n\nConvert spherical coordinates in degrees to Cartesian\ncoordinates (x, y, z)."},
  {"cart2sph", py_cart2sph, METH_VARARGS, "cart2sph(x, y, z) -> tuple[float, float, float]\n\nConvert Cartesian coordinates to spherical coordinates\n(lon, lat, magnitude), with angular outputs in degrees."},
  {NULL, NULL, 0, NULL}
};


/* Module definition */
static struct PyModuleDef spherical_functions = {
  PyModuleDef_HEAD_INIT,
  "spherical_functions",
  "Coordinate conversion helpers for spherical geometry.",
  -1,
  sphericalMethods
};

PyMODINIT_FUNC PyInit_spherical_functions(void)
{
    return PyModule_Create(&spherical_functions);
}