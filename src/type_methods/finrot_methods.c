#include <numpy/arrayobject.h>

#include "../types/finrot.h"
#include "../parse_number.c"
#include "covariance_methods.c"

// Function to convert Covariance to a matrix
gsl_matrix * cov_to_matrix(FiniteRot *fr_sph) {
    Covariance *cov = &fr_sph->Covariance;

    //double scalar = 2.0; // Scalar value
    // Multiply all elements in the matrix by the scalar
    //gsl_matrix_scale(matrix, scalar);

    return to_matrix(cov);
}

gsl_matrix * build_fr_ensemble(FiniteRot *fr_sph) {

    // ** If covariance is not given

    gsl_matrix *matrix;
    matrix = cov_to_matrix(fr_sph);

    return matrix;

    // ** Take action if covariance-matrix has negative or imaginary eigenvalues
    
    //correlated_ensemble_3d(const gsl_matrix *matrix, int Nsize)
}


static PyObject *py_build_fr_ensemble(PyObject *self, PyObject *args) {
    FiniteRot *fr_sph = (FiniteRot*)self;
    gsl_matrix *result;
    int n_size;


    if (PyTuple_Size(args) != 1){
        PyErr_SetString(PyExc_TypeError, "build_ensemble() expects a single argument, the number of samples to draw from the covariance matrix");
        return NULL; 
    }

    if (!PyArg_ParseTuple(args, "i", &n_size)) {
        return NULL;
    }
    

    char message[120];
    sprintf(message, "parsed n_size: %d\n", n_size);
    PySys_WriteStdout(message);

    

    /* result = build_fr_ensemble(fr_sph);

    npy_intp dims[2] = {result->size1, result->size2};
    PyObject *np_array = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, result->data);

    gsl_matrix_free(result);
    PyMem_Free(dims); 

    return np_array;*/
    return Py_BuildValue("i", result);
}