#include <Python.h>
#include <numpy/arrayobject.h>


static gsl_matrix * pyarray_to_gslmatrix(PyObject *pyarray) {
    if (!PyArray_Check(pyarray)) {
        PyErr_SetString(PyExc_TypeError, "pyarray_to_gslmatrix() expects an array as input");
        return NULL;
    }

    PyArrayObject* np_array = (PyArrayObject*)pyarray;

    if (PyArray_NDIM(np_array) != 2) {
        PyErr_SetString(PyExc_TypeError, "pyarray_to_gslmatrix() expects a 2D array as input");
        return NULL;
    }

    if (PyArray_TYPE(np_array) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "Input array must be of type double");
        return NULL;
    }

    npy_intp *dims = PyArray_DIMS(np_array);
    gsl_matrix *gsl_m = gsl_matrix_alloc(dims[0], dims[1]);
    
    for (int i = 0; i < dims[0]; i++) {
        for (int j = 0; j < dims[1]; j++) {
            double* ptr = (double*)PyArray_GETPTR2(np_array, i, j);
            double value = *ptr;
            gsl_matrix_set(gsl_m, i, j, value);
        }
    }
    return gsl_m;
}

static double* parse_tuple(PyObject *py_obj) {
    if (!PyTuple_Check(py_obj)) {
        PyErr_SetString(PyExc_TypeError, "Input must be a tuple");
        return NULL;
    }

    Py_ssize_t size = PyTuple_Size(py_obj);

    double *output_list = (double *)malloc((size + 1) * sizeof(double));
    output_list[0] = (double)size;

    if (output_list == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Memory allocation failed");
        return NULL;
    }

    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PyTuple_GetItem(py_obj, i);
        if (!PyFloat_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "Tuple must contain only float values");
            free(output_list);
            return NULL;
        }

        output_list[i + 1] = PyFloat_AsDouble(item);
    }   

    return output_list;
}

static double* parse_list(PyObject *py_obj) {
    if (!PyList_Check(py_obj)) {
        PyErr_SetString(PyExc_TypeError, "Input must be a list");
        return NULL;
    }

    Py_ssize_t size = PyList_Size(py_obj);

    double *output_list = (double *)malloc((size + 1) * sizeof(double));
    output_list[0] = (double)size;

    if (output_list == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Memory allocation failed");
        return NULL;
    }

    for (Py_ssize_t i = 0; i < size; ++i) {
        PyObject *item = PyList_GetItem(py_obj, i);
        if (!PyFloat_Check(item)) {
            PyErr_SetString(PyExc_TypeError, "List must contain only float values");
            free(output_list);
            return NULL;
        }

        output_list[i + 1] = PyFloat_AsDouble(item);
    }   

    return output_list;
}

static double* parse_numpy_1Darray(PyObject *py_obj) {
    Py_buffer buffer;
    double* output_list;
    int n;

    if (PyObject_GetBuffer(py_obj, &buffer, PyBUF_ANY_CONTIGUOUS | PyBUF_FORMAT) == -1){
        return NULL;
    }
    
    /* Check the dimensions of the array */
    if (buffer.ndim != 1){
        PyErr_SetString(PyExc_TypeError, "Input array must be one-dimensional");
        PyBuffer_Release(&buffer);
        return NULL;
    }
  
    /* Check the type of items in the array */
    if (strcmp(buffer.format, "d") != 0){
        PyErr_SetString(PyExc_TypeError, "Input array must contain float data");
        PyBuffer_Release(&buffer);
        return NULL;
    }

    n = (int)buffer.shape[0];
    double *data = (double *)buffer.buf;
    PyBuffer_Release(&buffer);

    output_list = (double *)malloc((n + 1) * sizeof(double));
    output_list[0] = (double)n;

    for (int i = 0; i < n; ++i) {
        output_list[i + 1] = data[i];
    }

    return output_list;
}

static double* parse_double_array(PyObject *py_obj) {

    // Check if tuple
    if (PyTuple_Check(py_obj)) {
        return parse_tuple(py_obj);
    
    // Check if list
    } else if (PyList_Check(py_obj)) {
        return parse_list(py_obj);

    // Check if numpy array
    } else {
        double *output_list = parse_numpy_1Darray(py_obj);
        if (output_list != NULL) {
            return output_list;

        } else { 
            PyErr_SetString(PyExc_TypeError, "Input must be a list, tuple, or numpy array");
            return NULL;
        }
    }
}


