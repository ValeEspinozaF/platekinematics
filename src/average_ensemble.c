#include "ensemble_methods.h"
#include "average_finrot.h"
#include "average_eulervec.h"

PyTypeObject CovarianceType;
PyTypeObject FiniteRotationType;
PyTypeObject EulerVectorType;


void average_vector(gsl_matrix* m_cart, double *v_cart, double *v_cov) {

    int N = (int)m_cart->size1;
    gsl_vector *x = gsl_vector_alloc(m_cart->size2);
    gsl_vector *y = gsl_vector_alloc(m_cart->size2);
    gsl_vector *z = gsl_vector_alloc(m_cart->size2);
    gsl_matrix_get_row(x, m_cart, 0);
    gsl_matrix_get_row(y, m_cart, 1);
    gsl_matrix_get_row(z, m_cart, 2);


    double x_sum = 0.0;
    double y_sum = 0.0;
    double z_sum = 0.0;


    for (int i = 0; i < N; i++) {
        x_sum += gsl_vector_get(x, i);
        y_sum += gsl_vector_get(y, i);
        z_sum += gsl_vector_get(z, i);
    }

    double x_mean = x_sum / N;
    double y_mean = y_sum / N;
    double z_mean = z_sum / N;

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

    for (int i = 0; i < N; i++) {
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

    v_cov[0] = x_squared_sum / N - x_mean * x_mean;
    v_cov[1] = xy_sum / N - x_mean * y_mean;
    v_cov[2] = xz_sum / N - x_mean * z_mean;
    v_cov[3] = y_squared_sum / N - y_mean * y_mean;
    v_cov[4] = yz_sum / N - y_mean * z_mean;
    v_cov[5] = z_squared_sum / N - z_mean * z_mean;
}


/* Methods contained in the module */
static PyMethodDef methodsMethods[] = {
  {"average_ev", py_ev_average, METH_VARARGS, "Return the average vector from a given ensemble of x, y and z vector coordinates in [units]."},
  {"average_fr", py_fr_average, METH_VARARGS, "Return the average finite rotation from a given ensemble of rotation matrices [units]."},
  {NULL, NULL, 0, NULL}
};


/* Module definition */
static struct PyModuleDef methods = {
  PyModuleDef_HEAD_INIT,
  "methods",
  "Custom methods module",
  -1,
  methodsMethods
};

PyMODINIT_FUNC PyInit_methods(void){
    PyObject *m;
    import_array(); // Initialize NumPy API

    if (PyType_Ready(&CovarianceType) < 0)
        return NULL;

    if (PyType_Ready(&EulerVectorType) < 0)
        return NULL;

    m = PyModule_Create(&methods);
    if (m == NULL)
        return NULL;

    Py_INCREF(&CovarianceType);
    PyModule_AddObject(m, "Covariance", (PyObject *)&CovarianceType);

    Py_INCREF(&EulerVectorType);
    PyModule_AddObject(m, "EulerVector", (PyObject *)&EulerVectorType);

    Py_INCREF(&FiniteRotationType);
    PyModule_AddObject(m, "EulerVector", (PyObject *)&FiniteRotationType);
    
    return m;
}