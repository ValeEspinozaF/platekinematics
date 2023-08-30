#define _USE_MATH_DEFINES
#include <Python.h>
#include <math.h>
#include "../gsl.h"
#include "../types/eulervec.h"
//#include "../spherical_functions.c"
//#include "covariance_methods.c"
//#include "../build_ensemble.c"


// Convert an Euler vector covariance [radians²/Myr²] to a 3x3 symmetric Matrix [degrees²/Myr²]. 
gsl_matrix * ev_cov_to_matrix(EulerVector *ev_sph) {
    if (!ev_sph->has_covariance) {
        PySys_WriteStdout("ev_cov_to_matrix.EulerVector must have a Covariance attribute\n");
        return NULL;
    }

    Covariance *cov = &ev_sph->Covariance;
    gsl_matrix *matrix = to_matrix(cov);

    double scalar = (180.0 / M_PI) * (180.0 / M_PI);
    gsl_matrix_scale(matrix, scalar);

    return matrix;
}


// Draw n rotation matrix samples from the covariance of a given finite rotation.
PyObject * build_ev_ensemble(EulerVector *ev_sph, int n_size) {
    double * ev_cart, *_ev_sph;
    gsl_matrix *cov_matrix;
    gsl_matrix *correlated_ens;
    
    if (!ev_sph->has_covariance){
        PyErr_SetString(PyExc_TypeError, "EulerVector must have a Covariance attribute");
        return NULL;
    }

    cov_matrix = ev_cov_to_matrix(ev_sph);

    // ** Take action if covariance-matrix has negative or imaginary eigenvalues

    correlated_ens = correlated_ensemble_3d(cov_matrix, n_size);
    ev_cart = sph2cart(ev_sph->Lon, ev_sph->Lat, ev_sph->AngVelocity);

    for (size_t i = 0; i < 3; i++) {
        for(size_t j = 0; j < n_size; j++) {
            double current_value = gsl_matrix_get(correlated_ens, i, j);
            gsl_matrix_set(correlated_ens, i, j, current_value + ev_cart[i]);
        }
    }


    PyObject* ev_ens = PyList_New(n_size);
    if (ev_ens == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create Python list");
        return NULL;
    }
    
    for (int i = 0; i < n_size; ++i) {
        EulerVector *ev = (EulerVector *)PyObject_New(EulerVector, Py_TYPE(ev_sph));

        if (ev != NULL) {
            _ev_sph = cart2sph(
                gsl_matrix_get(correlated_ens, 0, i), 
                gsl_matrix_get(correlated_ens, 1, i),
                gsl_matrix_get(correlated_ens, 2, i));

            ev->Lon = _ev_sph[0];
            ev->Lat = _ev_sph[1];
            ev->AngVelocity = _ev_sph[2];
            ev->TimeRange[0] = ev_sph->TimeRange[0];
            ev->TimeRange[1] = ev_sph->TimeRange[1];
            PyList_SET_ITEM(ev_ens, i, (PyObject*)ev);
            ev->has_covariance = 0;


        } else {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create EulerVector instance");
            Py_DECREF(ev_ens);
            return NULL;
        }
    }

    gsl_matrix_free(cov_matrix);
    gsl_matrix_free(correlated_ens);
    return ev_ens;
}


static PyObject *py_build_ev_ensemble(PyObject *self, PyObject *args) {
    EulerVector *ev_sph = (EulerVector *)self;
    PyObject *input_obj;
    double float_value;
    int n_size;
    

    if (PyTuple_Size(args) != 1){
        PyErr_SetString(PyExc_TypeError, "build_ensemble() expects a single argument, the number of samples to draw from the covariance matrix");
        return NULL; 
    }

    if (!PyArg_ParseTuple(args, "O", &input_obj)) {
        return NULL;
    }

    if (PyLong_Check(input_obj)) {
        PyArg_ParseTuple(args, "i", &n_size);
    } else if (PyFloat_Check(input_obj)) {
        PyArg_ParseTuple(args, "d", &float_value);
        n_size = (int)float_value;
    } else {
        Py_DECREF(input_obj);
        PyErr_SetString(PyExc_TypeError, "build_ensemble() expects an integer or a parsable float");
    }  
    Py_DECREF(input_obj);

    return build_ev_ensemble(ev_sph, n_size);
}