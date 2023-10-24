#include "platekinematics.h"

static bool check_covariance_eigs(gsl_vector *eig_va){
    bool cov_switch = true;

    double sum1 = 0.0;
    double sum2 = 0.0;

    for (size_t i = 0; i < eig_va->size; i++) {
        sum1 += gsl_vector_get(eig_va, i);
        sum2 += fabs(gsl_vector_get(eig_va, i));
    }

    double chk = sum1 / sum2;

    if (chk < 1){

        int negIdx[] = {0, 0, 0}; 
        int negCount = 0;

        for (size_t i = 0; i < eig_va->size; i++) {
            if (gsl_vector_get(eig_va, i) < 0) {
                negIdx[i] = 1;
                negCount++;
            }
        }

        if (negCount == 3) {
            cov_switch = false;

        } else {
            double max_eig_va = gsl_vector_max(eig_va);

            for (int i = 0; i < eig_va->size; i++) {
                if (negIdx[i] == 1){
                    double rel = fabs(gsl_vector_get(eig_va, i)) / max_eig_va;

                    if (rel > 0.05) {
                        cov_switch = false;
                        break;
                    }
                }
            }
        }
    }   

    return cov_switch;
}


gsl_matrix* correlated_ensemble_3d(gsl_matrix *cov_matrix, int n_size) {
    const size_t dim = 3;

    gsl_matrix *A = gsl_matrix_alloc(dim, dim);
	gsl_matrix_memcpy(A, cov_matrix);
    

    // Calculate eigenvalues and eigenvectors of the covariance matrix
    gsl_vector *eig_va = gsl_vector_alloc(dim);
    gsl_matrix *eig_ve = gsl_matrix_alloc(dim, dim);
    gsl_vector_complex *v = gsl_vector_complex_alloc(dim);

    gsl_eigen_symmv_workspace *workspace = gsl_eigen_symmv_alloc(dim);
    gsl_eigen_symmv(A, eig_va, eig_ve, workspace);


    if (check_covariance_eigs(eig_va) == false) {
        gsl_vector_free(eig_va);
        gsl_matrix_free(eig_ve);
        gsl_matrix_free(A);
        PyErr_SetString(PyExc_ValueError, "input Covariance yields all negative or too large negative eigenvalues");
        return NULL;
    }    

    gsl_rng *rng = gsl_rng_alloc(gsl_rng_default);

    // Use random seed each run
    time_t current_time = time (0);
    gsl_rng_set(rng, (unsigned long int) current_time);


    // Generate random data based on eigenvalues
    double sqrt_x = sqrt(gsl_vector_get(eig_va, 0));
    double sqrt_y = sqrt(gsl_vector_get(eig_va, 1));
    double sqrt_z = sqrt(gsl_vector_get(eig_va, 2));
    
    gsl_matrix *data = gsl_matrix_alloc(dim, n_size);
    for (int i = 0; i < n_size; i++) {
        gsl_matrix_set(data, 0, i, sqrt_x * gsl_ran_ugaussian(rng));
        gsl_matrix_set(data, 1, i, sqrt_y * gsl_ran_ugaussian(rng));
        gsl_matrix_set(data, 2, i, sqrt_z * gsl_ran_ugaussian(rng));
    }


    // Multiply eigen vectors with data matrix
    gsl_matrix *ndata = gsl_matrix_alloc(dim, n_size);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, eig_ve, data, 0.0, ndata);

    gsl_rng_free(rng);
    gsl_matrix_free(data);
    gsl_vector_free(eig_va);
    gsl_matrix_free(eig_ve);
    gsl_matrix_free(A);

    return ndata;
}