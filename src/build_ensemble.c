#define WIN32
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <stdio.h>
#include "types/finite_rotation.h"
#include "type_methods/covariance_methods.c"

/* GSL includes */
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_randist.h>

#include <numpy/arrayobject.h>

//void cov_to_matrix();

/* 
void correlated_ensemble_3d(const gsl_matrix *matrix, int Nsize) {
    gsl_vector *eigenvalues = gsl_vector_alloc(matrix->size1);
    gsl_matrix *eigenvectors = gsl_matrix_alloc(matrix->size1, matrix->size2);
    
    // Perform eigenvalue decomposition
    gsl_eigen_symmv_workspace *workspace = gsl_eigen_symmv_alloc(matrix->size1);
    gsl_eigen_symmv(matrix, eigenvalues, eigenvectors, workspace);
    gsl_eigen_symmv_free(workspace);

    // Generate correlated random data
    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);
    gsl_matrix *data = gsl_matrix_alloc(3, Nsize);
    gsl_matrix *ndata = gsl_matrix_alloc(3, Nsize);

    for (int i = 0; i < Nsize; i++) {
        for (int j = 0; j < 3; j++) {
            double val = gsl_ran_ugaussian(rng) * sqrt(gsl_vector_get(eigenvalues, j));
            gsl_matrix_set(data, j, i, val);
        }
    }

    // Transform the data using eigenvectors
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, eigenvectors, data, 0.0, ndata);

    gsl_vector_free(eigenvalues);
    gsl_matrix_free(eigenvectors);
    gsl_rng_free(rng);
    gsl_matrix_free(data);
    gsl_matrix_free(ndata);
} 



gsl_matrix * build_fr_ensemble(struct FiniteRotSph *FRs) {

    // ** If covariance is not given

    gsl_matrix *matrix;
    matrix = cov_to_matrix(FRs);

    // Print the transformed data
    printf("Transformed data:\n");
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            printf("%.4f ", gsl_matrix_get(matrix, j, i));
        }
        printf("\n");
    }

    return matrix;

    // ** Take action if covariance-matrix has negative or imaginary eigenvalues
    
    //correlated_ensemble_3d(const gsl_matrix *matrix, int Nsize)
}


static PyObject *py_build_fr_ensemble(PyObject *self, PyObject *args) {
    struct FiniteRotSph *FRs;
    size_t size;
    gsl_matrix *result;

    if (!PyArg_ParseTuple(args, "w#", &FRs, &size))
        return NULL;

    if (size != sizeof(struct FiniteRotSph)) {
        PyErr_SetString(PyExc_TypeError, "wrong buffer size");
        return NULL;
    }
  

    result = build_fr_ensemble(FRs);

    npy_intp dims[2] = {result->size1, result->size2};
    PyObject *np_array = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, result->data);

    gsl_matrix_free(result);
    PyMem_Free(dims);

    return np_array;
}





static PyMethodDef build_ensemble_methods[] = {
    {"build_fr_ensemble", py_build_fr_ensemble, METH_VARARGS, "Draws n rotation matrix samples from the covariance of a given finite rotation."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef build_ensemble = {
    PyModuleDef_HEAD_INIT,
    "build_ensemble",
    NULL,
    -1,
    build_ensemble_methods
};

PyMODINIT_FUNC PyInit_build_ensemble(void) {
    import_array(); // Initialize NumPy API
    return PyModule_Create(&build_ensemble);
}

*/