#include <Python.h>

/* Main function */
int square(int num) {
    return num * num;
}

float times2(float num) {
    return num * 2;
}

static PyObject *py_square(PyObject *self, PyObject *args) {

  /* Declare variables */
  int n_num, result;

  /* Parse argument from python to local variable (n_num) */
  if (!PyArg_ParseTuple(args, "i", &n_num)) {
    return NULL;
  }

  /* Assign value to output variable */
  result = square(n_num);

  /* Return */
  return Py_BuildValue("i", result);
}


static PyObject *py_times2(PyObject *self, PyObject *args) {

  /* Declare variables */
  float n_num, result;

  /* Parse argument from python to local variable (n_num) */
  if (!PyArg_ParseTuple(args, "f", &n_num)) {
    return NULL;
  }

  /* Assign value to output variable */
  result = times2(n_num);

  /* Return */
  return Py_BuildValue("f", result);
}


/* Methods contained in the module */
static PyMethodDef mathsMethods[] = {
  {"square", py_square, METH_VARARGS, "Function for calculating square in C"},
  {"times2", py_times2, METH_VARARGS, "Function for calculating two times in C"},
  {NULL, NULL, 0, NULL}
};


/* Module definition */
static struct PyModuleDef maths = {
  PyModuleDef_HEAD_INIT,
  "maths",
  "Custom maths module",
  -1,
  mathsMethods
};

PyMODINIT_FUNC PyInit_maths(void)
{
    return PyModule_Create(&maths);
}
