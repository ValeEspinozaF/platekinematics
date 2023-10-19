#include "ensemble_methods.h"

void average_vector(gsl_matrix* m_cart, double *v_cart, double *v_cov) {

    int n_size = (int)m_cart->size2;
    gsl_vector *x = gsl_vector_alloc(n_size);
    gsl_vector *y = gsl_vector_alloc(n_size);
    gsl_vector *z = gsl_vector_alloc(n_size);
    gsl_matrix_get_row(x, m_cart, 0);
    gsl_matrix_get_row(y, m_cart, 1);
    gsl_matrix_get_row(z, m_cart, 2);
    PySys_WriteStdout("x0, x1, x2: %f, %f, %f\n", gsl_vector_get(x, 0), gsl_vector_get(x, 1), gsl_vector_get(x, 2));

    double x_sum = 0.0;
    double y_sum = 0.0;
    double z_sum = 0.0;


    for (int i = 0; i < n_size; i++) {
        x_sum += gsl_vector_get(x, i);
        y_sum += gsl_vector_get(y, i);
        z_sum += gsl_vector_get(z, i);
    }

    double x_mean = x_sum / n_size;
    double y_mean = y_sum / n_size;
    double z_mean = z_sum / n_size;

    v_cart[0] = x_mean;
    v_cart[1] = y_mean;
    v_cart[2] = z_mean;


    // Calculate covariance elements [units²]
    double x_squared_sum = 0.0;
    double y_squared_sum = 0.0;
    double z_squared_sum = 0.0;
    double xy_sum = 0.0;
    double xz_sum = 0.0;
    double yz_sum = 0.0;

    for (int i = 0; i < n_size; i++) {
        double xv = gsl_vector_get(x, i);
        double yv = gsl_vector_get(y, i);
        double zv = gsl_vector_get(z, i);

        x_squared_sum += xv * xv;
        y_squared_sum += yv * yv;
        z_squared_sum += zv * zv;
        xy_sum += xv * yv;
        xz_sum += xv * zv;
        yz_sum += yv * zv;
    }

    v_cov[0] = x_squared_sum/n_size - x_mean*x_mean;
    v_cov[1] = xy_sum/n_size - x_mean*y_mean;
    v_cov[2] = xz_sum/n_size - x_mean*z_mean;
    v_cov[3] = y_squared_sum/n_size - y_mean*y_mean;
    v_cov[4] = yz_sum/n_size - y_mean*z_mean;
    v_cov[5] = z_squared_sum/n_size - z_mean*z_mean;

    gsl_vector_free(x);
    gsl_vector_free(y);
    gsl_vector_free(z);
}