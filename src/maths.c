#include <Python.h>

/* Main function */
inline int square(int num) {
    return num * num;
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


double avg(double *a, int n)
{
    int i;
    double total = 0.0;
    for (i = 0; i < n; i++)
    {
        total += a[i];
    }
    return total / n;
}

static PyObject *py_avg(PyObject *self, PyObject *args)
{
    PyObject *bufobj;
    Py_buffer view;
    double result;

    /* Get the passed Python object */
    if (!PyArg_ParseTuple(args, "O", &bufobj)){
        return NULL;
    }
  
    /* Attempt to extract buffer information from it */
    if (PyObject_GetBuffer(bufobj, &view, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1){
        return NULL;
    }
  
    if (view.ndim != 1){
        PyErr_SetString(PyExc_TypeError, "Expected a 1-dimensional array");
        PyBuffer_Release(&view);
        return NULL;
    }
  
    /* Check the type of items in the array */
    if (strcmp(view.format, "d") != 0){
        PyErr_SetString(PyExc_TypeError, "Expected an array of doubles");
        PyBuffer_Release(&view);
        return NULL;
    }
  
    /* Pass the raw buffer and size to the C function */
    result = avg(view.buf, view.shape[0]);
  
    /* Indicate we're done working with the buffer */
    PyBuffer_Release(&view);
    return Py_BuildValue("d", result);
}


/* Methods contained in the module */
static PyMethodDef mathsMethods[] = {
  {"square", py_square, METH_VARARGS, "Function for calculating square in C"},
  {"avg", py_avg, METH_VARARGS, "Get the average of a numpy.array."},
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
