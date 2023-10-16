#include "type_methods.h"

PyObject* build_numpy_1Darray(gsl_vector *cA);

gsl_vector* cov_to_gsl_vector(Covariance *cov) {
    
    gsl_vector *cov_vector = gsl_vector_alloc(6);
    gsl_vector_set(cov_vector, 0, cov->C11);
    gsl_vector_set(cov_vector, 1, cov->C12);
    gsl_vector_set(cov_vector, 2, cov->C13);
    gsl_vector_set(cov_vector, 3, cov->C22);
    gsl_vector_set(cov_vector, 4, cov->C23);
    gsl_vector_set(cov_vector, 5, cov->C33);    
    return cov_vector;
}

PyObject *py_cov_to_numpy(PyObject *self, int Py_UNUSED(_)) {
    Covariance *cov = (Covariance*)self;
    gsl_vector* cov_vector = cov_to_gsl_vector(cov);
    return build_numpy_1Darray(cov_vector);
}