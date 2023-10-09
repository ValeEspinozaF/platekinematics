#include "ensemble_methods.h"

void average_vector(gsl_matrix* m_cart, double *v_cart, double *v_cov);


static gsl_matrix** pylist_fr_to_gslmatrix(PyObject *py_array) {
    int n_size = (int)PyList_Size(py_array);
    gsl_matrix** m_array = (gsl_matrix**)malloc(n_size * sizeof(gsl_matrix*));

    for (int i = 0; i < n_size; i++) {
        m_array[i] = gsl_matrix_alloc(3, 3);

        PyObject *fr_pyob = PyList_GetItem(py_array, i);
        /* if (!PyObject_TypeCheck(ev_pyob, &FiniteRotType)) {
            PySys_WriteStdout("Item %d is not a FiniteRot instance\n", i);
        } */ //FIXME Not working whyyy

        FiniteRot *fr = (FiniteRot *)fr_pyob;
        m_array[i] = fr_to_rotation_matrix(fr); // can i just set them all at once?
    }

    return m_array;
}


static PyObject *py_fr_average(PyObject *self, PyObject *args) {
    int n_args;
    PyObject *fr_pyob;
    double time = 0.0;

    PyObject *original_type, *original_value, *original_traceback; 

    
    n_args = (int)PyTuple_Size(args);
    if (n_args == 1 || n_args == 2) {
        if (!PyArg_ParseTuple(args, "O|d", &fr_pyob, &time)) {
            PyErr_SetString(PyExc_TypeError, "average() expects two arguments, a numpy array with the rotation matrix ensemble and float with the age of the ensemble");
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "average() expects two arguments, a numpy array with the rotation matrix ensemble and float with the age of the ensemble");
        return NULL; 
    } 


    // Ensemble arg parsing
    gsl_matrix** fr_gsl;
    if (PyArray_Check(fr_pyob)) {
        fr_gsl = pyarray3D_to_gslmatrix(fr_pyob);
        if (fr_gsl == NULL) {
            PyErr_Fetch(&original_type, &original_value, &original_traceback);
            PyErr_Restore(original_type, original_value, original_traceback);
            return NULL;
        }

    } else if (PyList_Check(fr_pyob)) {
        fr_gsl = pylist_fr_to_gslmatrix(fr_pyob);
        if (fr_gsl == NULL) {
            PyErr_Fetch(&original_type, &original_value, &original_traceback);
            PyErr_Restore(original_type, original_value, original_traceback);
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "average() expects a numpy array as input");
        return NULL;
    }
    

    // Calculate average finite rotation from GSL matrix
    gsl_matrix* ea_array = rotation_matrices_to_eas(fr_gsl);
    if (ea_array == NULL) { //!!! Function does not throw any errors
        PyErr_Fetch(&original_type, &original_value, &original_traceback);
        PyErr_Restore(original_type, original_value, original_traceback);
        return NULL;
    }

    int n_size = (int)(sizeof(fr_gsl) / sizeof(fr_gsl[0]));
    for (int i = 0; i < n_size; i++) {
        gsl_matrix_free(fr_gsl[i]);
    }   

    double* ea_cart = (double*)malloc(3 * sizeof(double));
    double* fr_cov = (double*)malloc(6 * sizeof(double));
    
    average_vector(ea_array, ea_cart, fr_cov);
    gsl_matrix_free(ea_array);

    FiniteRot* fr_avg = ea_to_finrot(ea_cart);
    if (fr_avg != NULL) {
        fr_avg->Time = time;

        Covariance *cov = PyObject_New(Covariance, &CovarianceType);
        if (cov != NULL) {
            cov->C11 = fr_cov[0] * (M_PI / 180.0) * (M_PI / 180.0);
            cov->C12 = fr_cov[1] * (M_PI / 180.0) * (M_PI / 180.0);
            cov->C13 = fr_cov[2] * (M_PI / 180.0) * (M_PI / 180.0);
            cov->C22 = fr_cov[3] * (M_PI / 180.0) * (M_PI / 180.0);
            cov->C23 = fr_cov[4] * (M_PI / 180.0) * (M_PI / 180.0);
            cov->C33 = fr_cov[5] * (M_PI / 180.0) * (M_PI / 180.0);
            fr_avg->Covariance = *cov; 
            fr_avg->has_covariance = 1;
            
        } else {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create Covariance instance");
            return NULL;
        }

    } else {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create EulerVector instance");
        return NULL;
    }

    return (PyObject *)fr_avg;
}