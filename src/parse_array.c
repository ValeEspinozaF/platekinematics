#include <Python.h>
#include <numpy/arrayobject.h>


static double* parse_numpy_array(PyObject *py_obj) {
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

    // Check if list/tuple
    if (PyList_Check(py_obj) || PyTuple_Check(py_obj)) {
        PyObject *input_list = PySequence_Fast(py_obj, "Input must be a sequence");
        if (input_list == NULL) {
            return NULL;
        }

        Py_ssize_t size = PyList_Size(input_list);

        
        double *output_list = (double *)malloc((size + 1) * sizeof(double));
        output_list[0] = (double)size;        

        if (output_list == NULL) {
            PyErr_SetString(PyExc_MemoryError, "Memory allocation failed");
            return NULL;
        }

        for (Py_ssize_t i = 0; i < size; ++i) {
            PyObject *item = PyList_GetItem(input_list, i);
            if (!PyFloat_Check(item)) {
                PyErr_SetString(PyExc_TypeError, "List must contain only float values");
                free(output_list);
                return NULL;
            }

            output_list[i + 1] = PyFloat_AsDouble(item);
        }   

        Py_DECREF(input_list);
        return output_list;


    // Check if numpy array
    } else {
        double *output_list = parse_numpy_array(py_obj);
        if (output_list != NULL) {
            return output_list;

        } else { 
            PyErr_SetString(PyExc_TypeError, "Input must be a list, tuple, or numpy array");
            return NULL;
        }
    }
}


