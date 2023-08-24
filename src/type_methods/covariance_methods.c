#include "covariance_methods.h"

// Function to convert Covariance to a matrix
gsl_matrix * to_matrix(Covariance *cov) {

    gsl_matrix *m = gsl_matrix_calloc(3, 3);

    gsl_matrix_set(m, 0, 0, cov->C11);
    gsl_matrix_set(m, 0, 1, cov->C12);
    gsl_matrix_set(m, 0, 2, cov->C13);
    gsl_matrix_set(m, 1, 0, cov->C13);
    gsl_matrix_set(m, 1, 1, cov->C22);
    gsl_matrix_set(m, 1, 2, cov->C23);
    gsl_matrix_set(m, 2, 0, cov->C23);
    gsl_matrix_set(m, 2, 1, cov->C23);
    gsl_matrix_set(m, 2, 2, cov->C33);
    return m;
}