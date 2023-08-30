#include <Python.h>

static void * parse_integer_or_float(PyObject *py_args, PyObject *py_obj, int *result) {
    double float_value;

    if (PyLong_Check(py_obj)) {
        PyArg_ParseTuple(py_args, "i", &result);
        PySys_WriteStdout("passed PyLong_Check\n");
    }

    else if (PyFloat_Check(py_obj)) {
        PyArg_ParseTuple(py_args, "d", &float_value);
        *result = (int)float_value;
        PySys_WriteStdout("passed PyFloat_Check\n");
    }

    else {
        PyErr_SetString(PyExc_TypeError, "Parse error: expected a PyObject integer or float");
    }

    return NULL;
} 