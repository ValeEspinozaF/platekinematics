#include "parse_array.h"


static gsl_matrix* pyarray2D_to_gslmatrix(PyObject *pyarray) {
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

static gsl_matrix** pyarray3D_to_gslmatrix(PyObject *pyarray) {
    if (!PyArray_Check(pyarray)) {
        PyErr_SetString(PyExc_TypeError, "pyarray_to_gslmatrix() expects an array as input");
        return NULL;
    }

    PyArrayObject* np_array = (PyArrayObject*)pyarray;

    if (PyArray_NDIM(np_array) != 3) {
        PyErr_SetString(PyExc_TypeError, "pyarray_to_gslmatrix() expects a 3D array as input");
        return NULL;
    }

    if (PyArray_TYPE(np_array) != NPY_DOUBLE) {
        PyErr_SetString(PyExc_TypeError, "Input array must be of type double");
        return NULL;
    }

    npy_intp *dims = PyArray_DIMS(np_array);
    gsl_matrix** m_array = (gsl_matrix**)malloc(dims[0] * sizeof(gsl_matrix*));
    
    for (int i = 0; i < dims[0]; i++) {
        m_array[i] = gsl_matrix_alloc(3, 3);

        for (int j = 0; j < dims[1]; j++) {
            for (int k = 0; k < dims[2]; j++) {
                double* ptr = (double*)PyArray_GETPTR3(np_array, i, j, k);
                double value = *ptr;
                gsl_matrix_set(m_array[i], j, k, value);
            }
        }
    }
    return m_array;
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

double* parse_double_array(PyObject *py_obj) {

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

PyObject* build_numpy_3Darray(gsl_matrix **cA, int dim0_n_size) {
    npy_intp dims[3] = {dim0_n_size, (int)cA[0]->size1, (int)cA[0]->size2};

    // Create a new NumPy array and copy data
    PyObject *np_array = PyArray_SimpleNew(3, dims, NPY_DOUBLE);
    double *np_data = (double *)PyArray_DATA((PyArrayObject *)np_array);

    for (int i = 0; i < dim0_n_size; ++i) {
        gsl_matrix *m = cA[i];
        for (int j = 0; j < m->size1; ++j) {
            for (int k = 0; k < m->size2; ++k) {
                np_data[i * m->size1 * m->size2 + j * m->size2 + k] = gsl_matrix_get(m, j, k);
            }
        }
        gsl_matrix_free(m);
    }
    
    return np_array;
}

PyObject* build_numpy_2Darray(gsl_matrix *cA) {
    npy_intp dims[2]; 
    dims[0] = (int)cA->size1;
    dims[1] = (int)cA->size2;

    // Create a new NumPy array and copy data
    PyObject *np_array = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    double *np_data = (double *)PyArray_DATA((PyArrayObject *)np_array);

    for (int i = 0; i < cA->size1; ++i) {
        for (int j = 0; j < cA->size2; ++j) {
            np_data[i * cA->size2 + j] = gsl_matrix_get(cA, i, j);
        }
    }

    gsl_matrix_free(cA);
    return np_array;
}