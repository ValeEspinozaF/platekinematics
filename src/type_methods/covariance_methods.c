#include "../types/finite_rotation.h"
#include <gsl/gsl_matrix.h>

// Function to convert Covariance to a matrix
gsl_matrix * cov_to_matrix(struct FiniteRot *FRs) {

    /*
    const struct Covariance *cov = NULL;
    cov = &fr_sph->Covariance;
    */

    Covariance *cov = &FRs->Covariance;

    // Create and populate the covariance matrix
    gsl_matrix *m = gsl_matrix_calloc(3, 3);
    
    /*
    double x = cov->C11;
    gsl_matrix_set(m, 0, 0, double x)
    */

    gsl_matrix_set(m, 0, 0, cov->C11);

    /*
    covMatrix[0][0] = cov->C11;
    covMatrix[0][1] = cov->C12;
    covMatrix[0][2] = cov->C13;
    covMatrix[1][0] = cov->C12;
    covMatrix[1][1] = cov->C22;
    covMatrix[1][2] = cov->C23;
    covMatrix[2][0] = cov->C13;
    covMatrix[2][1] = cov->C23;
    covMatrix[2][2] = cov->C33;
    */

   return m;
}