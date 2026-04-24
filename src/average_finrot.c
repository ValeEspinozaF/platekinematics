#define PY_ARRAY_UNIQUE_SYMBOL PLATEKIN_ARRAY_API // Must be defined before importing numpy/arrayobject.h
#define NO_IMPORT_ARRAY
#include "ensemble_methods.h"

void average_vector(gsl_matrix* m_cart, double *v_cart, double *v_cov);


PyObject *py_fr_average(PyObject *self, PyObject *args) {
    PyObject *original_type, *original_value, *original_traceback; 
    PyObject *fr_pyob;
    gsl_matrix** fr_gsl;
    int n_args;
    double time = 0.0;

    
    n_args = (int)PyTuple_Size(args);
    if (n_args == 1 || n_args == 2) {
        if (!PyArg_ParseTuple(args, "O|d", &fr_pyob, &time)) {
            PyErr_SetString(PyExc_TypeError, "average_fr() expects two arguments, a numpy array with the rotation matrix ensemble and float with the age of the ensemble");
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "average_fr() expects two arguments, a numpy array with the rotation matrix ensemble and float with the age of the ensemble");
        return NULL; 
    }
    
    int* ptr_dim0_n_size = (int*)malloc(sizeof(int));
    

    if (PyArray_Check(fr_pyob)) {
        fr_gsl = pyarray3D_to_gslmatrix(fr_pyob, &ptr_dim0_n_size);
        if (fr_gsl == NULL) {
            PyErr_Fetch(&original_type, &original_value, &original_traceback);
            PyErr_Restore(original_type, original_value, original_traceback);
            return NULL;
        }

    } else if (PyList_Check(fr_pyob)) {
        PyErr_SetString(PyExc_NotImplementedError, "average_fr() not implemented for lists");
        return NULL;
        
    } else {
        PyErr_SetString(PyExc_TypeError, "average_fr() expects a numpy array as input");
        return NULL;
    }

    // Calculate average finite rotation from GSL matrix
    gsl_matrix* ea_array = rotation_matrices_to_eas(fr_gsl, *ptr_dim0_n_size);

    for (int i = 0; i < *ptr_dim0_n_size; i++) {
        gsl_matrix_free(fr_gsl[i]);
    }   

    free(ptr_dim0_n_size);

   
    double* ea_cart = (double*)malloc(3 * sizeof(double));
    double* fr_cov = (double*)malloc(6 * sizeof(double));
    average_vector(ea_array, ea_cart, fr_cov);
    gsl_matrix_free(ea_array);

    FiniteRot* fr_avg = ea_to_finrot(ea_cart);
    free(ea_cart);

    if (fr_avg != NULL) {
        fr_avg->Time = time;
        fr_avg->Covariance.C11 = fr_cov[0] * (M_PI / 180.0) * (M_PI / 180.0);
        fr_avg->Covariance.C12 = fr_cov[1] * (M_PI / 180.0) * (M_PI / 180.0);
        fr_avg->Covariance.C13 = fr_cov[2] * (M_PI / 180.0) * (M_PI / 180.0);
        fr_avg->Covariance.C22 = fr_cov[3] * (M_PI / 180.0) * (M_PI / 180.0);
        fr_avg->Covariance.C23 = fr_cov[4] * (M_PI / 180.0) * (M_PI / 180.0);
        fr_avg->Covariance.C33 = fr_cov[5] * (M_PI / 180.0) * (M_PI / 180.0);
        fr_avg->has_covariance = 1;

    } else {
        free(fr_cov);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create FiniteRotation instance");
        return NULL;
    } 

    free(fr_cov);
    return (PyObject *)fr_avg;
}