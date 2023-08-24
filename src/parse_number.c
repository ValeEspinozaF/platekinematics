#include <Python.h>

/* static int parse_integer_or_float(PyObject *obj) {
    if (PyLong_Check(obj)) {
        return (int)PyLong_AsLong(obj); // Successfully parsed as integer 
    }

    if (PyFloat_Check(obj)) {
        return (int)PyFloat_AsDouble(obj); // Successfully parsed as float (converted to int)
    }

    return NULL;
} */

static int parse_integer_or_float(PyObject *obj, int *result) {
    if (PyLong_Check(obj)) {
        *result = PyLong_AsLong(obj);
        return 1;  // Successfully parsed as integer
    }

    if (PyFloat_Check(obj)) {
        *result = (int)PyFloat_AsDouble(obj);
        return 1;  // Successfully parsed as float (converted to int)
    }

    // Parsing failed
    return 0;
}