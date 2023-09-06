#define _USE_MATH_DEFINES
#include <math.h>
#include "covariance_methods.h"

// Function to convert covariance elements to a covariance matrix
gsl_matrix * to_matrix(Covariance *cov) {

    gsl_matrix *m = gsl_matrix_calloc(3, 3);

    gsl_matrix_set(m, 0, 0, cov->C11);
    gsl_matrix_set(m, 0, 1, cov->C12);
    gsl_matrix_set(m, 0, 2, cov->C13);
    gsl_matrix_set(m, 1, 1, cov->C22);
    gsl_matrix_set(m, 1, 2, cov->C23);
    gsl_matrix_set(m, 2, 2, cov->C33);

    gsl_matrix_set(m, 1, 0, cov->C12);
    gsl_matrix_set(m, 2, 0, cov->C13);
    gsl_matrix_set(m, 2, 1, cov->C23);

    return m;
}

// Convert an Euler vector covariance [radians²/Myr²] to a 3x3 symmetric Matrix [degrees²/Myr²]. 
gsl_matrix * ev_cov_to_matrix(EulerVector *ev_sph) {
    if (!ev_sph->has_covariance) {
        PySys_WriteStdout("EulerVector must have a Covariance attribute\n");
        return NULL;
    }

    Covariance *cov = &ev_sph->Covariance;
    gsl_matrix *matrix = to_matrix(cov);

    double scalar = (180.0 / M_PI) * (180.0 / M_PI);
    gsl_matrix_scale(matrix, scalar);

    return matrix;
}


// Convert a finite rotation covariance [radians²] to a 3x3 symmetric matrix [radians²].
gsl_matrix* fr_cov_to_matrix(FiniteRot *fr_sph) {
    Covariance *cov = &fr_sph->Covariance;
    return to_matrix(cov);
}