#define WIN32
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

//#include "parse_number.c"
//#include "types/finrot.h"
//#include <gsl/gsl_matrix.h>
//#include "type_methods/finrot_methods.c"


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
*/







/* static PyMethodDef build_ensemble_methods[] = {
    {"build_ensemble", py_build_fr_ensemble, METH_VARARGS, "Draws n FiniteRotation() samples from the covariance of a given finite rotation."},
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
} */