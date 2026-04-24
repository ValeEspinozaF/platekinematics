#define PY_ARRAY_UNIQUE_SYMBOL PLATEKIN_ARRAY_API // Must be defined before importing numpy/arrayobject.h
#define NO_IMPORT_ARRAY
#include "ensemble_methods.h"

void average_vector(gsl_matrix* m_cart, double *v_cart, double *v_cov);


static gsl_matrix** pylist_fr_to_gslmatrixarray(PyObject *py_array, int **dim0_n_size) {
    int n_size = (int)PyList_Size(py_array);
    gsl_matrix **fr_gsl = (gsl_matrix **)malloc(n_size * sizeof(gsl_matrix *));

    if (fr_gsl == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate rotation matrix array");
        return NULL;
    }

    **dim0_n_size = n_size;

    for (int i = 0; i < n_size; i++) {
        PyObject *fr_pyob = PyList_GetItem(py_array, i);
        if (!PyObject_TypeCheck(fr_pyob, &FiniteRotationType)) {
            for (int j = 0; j < i; j++) {
                gsl_matrix_free(fr_gsl[j]);
            }
            free(fr_gsl);
            PyErr_SetString(PyExc_TypeError, "Input list must contain only FiniteRotation objects");
            return NULL;
        }

        fr_gsl[i] = fr_to_rotation_matrix((FiniteRot *)fr_pyob);
        if (fr_gsl[i] == NULL) {
            for (int j = 0; j < i; j++) {
                gsl_matrix_free(fr_gsl[j]);
            }
            free(fr_gsl);
            PyErr_SetString(PyExc_RuntimeError, "Failed to convert FiniteRotation to rotation matrix");
            return NULL;
        }
    }

    return fr_gsl;
}


PyObject *py_fr_average(PyObject *self, PyObject *args) {
    PyObject *original_type, *original_value, *original_traceback; 
    PyObject *fr_pyob;
    gsl_matrix** fr_gsl;
    int n_args;
    double time = 0.0;

    
    n_args = (int)PyTuple_Size(args);
    if (n_args == 1 || n_args == 2) {
        if (!PyArg_ParseTuple(args, "O|d", &fr_pyob, &time)) {
            PyErr_SetString(PyExc_TypeError, "average_fr() expects one or two arguments: a list of FiniteRotation objects or a NumPy array of rotation matrices, and an optional float time");
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "average_fr() expects one or two arguments: a list of FiniteRotation objects or a NumPy array of rotation matrices, and an optional float time");
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
        fr_gsl = pylist_fr_to_gslmatrixarray(fr_pyob, &ptr_dim0_n_size);
        if (fr_gsl == NULL) {
            PyErr_Fetch(&original_type, &original_value, &original_traceback);
            PyErr_Restore(original_type, original_value, original_traceback);
            return NULL;
        }
        
    } else {
        PyErr_SetString(PyExc_TypeError, "average_fr() expects a list of FiniteRotation objects or a NumPy array of rotation matrices");
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