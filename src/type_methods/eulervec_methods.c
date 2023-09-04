#define _USE_MATH_DEFINES
#include <Python.h>
#include <math.h>
#include <stdbool.h>
#include <numpy/arrayobject.h>

#include "../gsl.h"
#include "../types/eulervec.h"


void capsule_cleanup_ev(PyObject *capsule) {
    void *memory = PyCapsule_GetPointer(capsule, NULL);
    free(memory);
}

PyObject * build_numpy_array(gsl_matrix *cA) {
    npy_intp dims[2]; 
    dims[0] = (int)cA->size1;
    dims[1] = (int)cA->size2;

    // Create a new NumPy array and copy data
    PyObject *np_array = PyArray_SimpleNew(2, dims, NPY_DOUBLE);
    double *np_data = (double *)PyArray_DATA((PyArrayObject *)np_array);

    for (int i = 0; i < cA->size1; ++i) {
        for (int j = 0; j < cA->size2; ++j) {
            np_data[i * cA->size2 + j] = gsl_matrix_get(cA, i, j);
        }
    }

    gsl_matrix_free(cA);
    return np_array;
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


gsl_matrix * build_ev_array(EulerVector *ev_sph, int n_size, const char* coordinate_system) {
    gsl_matrix *cov_matrix, *ev_ens;
    double * ev_cart, *_ev_sph;
    bool out_spherical;

    //test
    if (n_size == 5) {
        PyErr_SetString(PyExc_ValueError, "Nor a 5 for christ sake");
        return NULL;
    }

    if (strcmp(coordinate_system, "spherical") == 0) {
        out_spherical = true;

    } else if (strcmp(coordinate_system, "cartesian") == 0) {
        out_spherical = false;

    } else {
        PyErr_SetString(PyExc_TypeError, "Input coordinate_system assigns and invalid coordinate system. Valid options are spherical and cartesian");
        return NULL; 
    }
    

    if (ev_sph->has_covariance == 0){
        PyErr_SetString(PyExc_TypeError, "EulerVector must have a Covariance attribute");
        return NULL;
    }

    cov_matrix = ev_cov_to_matrix(ev_sph);
    if (cov_matrix == NULL) {
        PyErr_SetString(PyExc_TypeError, "Failed transforming covariance elements to matrix\n");
        return NULL;
    }

    // ** Take action if covariance-matrix has negative or imaginary eigenvalues
    
    ev_cart = sph2cart(ev_sph->Lon, ev_sph->Lat, ev_sph->AngVelocity);

    ev_ens = correlated_ensemble_3d(cov_matrix, n_size);
    if (ev_ens == NULL) {
        gsl_matrix_free(cov_matrix);
        PyErr_SetString(PyExc_MemoryError, "Failed to create correlated ensemble matrix");
        return NULL;
    }  

    gsl_matrix_free(cov_matrix);  
/*     gsl_matrix *ev_ens = gsl_matrix_alloc(3, n_size);
    gsl_matrix_memcpy(ev_ens, correlated_ens);
    gsl_matrix_free(correlated_ens); */


    for(size_t i = 0; i < n_size; i++) {
        double cx, cy, cz;
        cx = gsl_matrix_get(ev_ens, 0, i) + ev_cart[0];
        cy = gsl_matrix_get(ev_ens, 1, i) + ev_cart[1];
        cz = gsl_matrix_get(ev_ens, 2, i) + ev_cart[2];
        if (out_spherical) {
            _ev_sph = cart2sph(cx, cy, cz);
            gsl_matrix_set(ev_ens, 0, i, _ev_sph[0]);
            gsl_matrix_set(ev_ens, 1, i, _ev_sph[1]);
            gsl_matrix_set(ev_ens, 2, i, _ev_sph[2]);
        } else {
            gsl_matrix_set(ev_ens, 0, i, cx);
            gsl_matrix_set(ev_ens, 1, i, cy);
            gsl_matrix_set(ev_ens, 2, i, cz);
        }
    }
    return build_numpy_array(ev_ens);
}


// Draw n rotation matrix samples from the covariance of a given finite rotation.
PyObject * build_ev_ensemble(EulerVector *ev_sph, int n_size) {
    const char *coordinate_system = "spherical";
    gsl_matrix *ev_array; 

    ev_array = build_ev_array(ev_sph, n_size, coordinate_system);

    // Fetch potential errors from build_ev_array
    PyObject *original_type, *original_value, *original_traceback; 
    PyErr_Fetch(&original_type, &original_value, &original_traceback);
    if (ev_array == NULL) {
        PyErr_Restore(original_type, original_value, original_traceback);
        return NULL;
    }


    PyObject* ev_ens = PyList_New(n_size);
    if (ev_ens == NULL) {
        PyErr_SetString(PyExc_MemoryError, "Failed to create Python list");
        return NULL;
    }
    
    for (int i = 0; i < n_size; ++i) {
        EulerVector *ev = (EulerVector *)PyObject_New(EulerVector, Py_TYPE(ev_sph));
        if (ev != NULL) {
            ev->Lon = gsl_matrix_get(ev_array, 0, i);
            ev->Lat = gsl_matrix_get(ev_array, 1, i);
            ev->AngVelocity = gsl_matrix_get(ev_array, 2, i);
            ev->TimeRange[0] = ev_sph->TimeRange[0];
            ev->TimeRange[1] = ev_sph->TimeRange[1];
            ev->has_covariance = 0;
            PyList_SET_ITEM(ev_ens, i, (PyObject*)ev);

        } else {
            PyErr_SetString(PyExc_RuntimeError, "Failed to create EulerVector instance");
            Py_DECREF(ev_ens);
            return NULL;
        }
    }

    gsl_matrix_free(ev_array);
    return ev_ens;
}


static PyObject *py_build_ev_array(PyObject *self, PyObject *args) {
    EulerVector *ev_sph = (EulerVector *)self;
    PyObject *n_size_obj = NULL;
    int n_size, n_args;
    const char* coordinate_system = NULL;
    
    n_args = (int)PyTuple_Size(args);
    if (n_args == 1 || n_args == 2) {
        if (!PyArg_ParseTuple(args, "O|s", &n_size_obj, &coordinate_system)) {
            PyErr_SetString(PyExc_TypeError, "build_array() expects one or two arguments, an integer with the number of samples and a string assigning the coordinate system of the output matrix");
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "build_array() expects one or two arguments, an integer with the number of samples and a string assigning the coordinate system of the output matrix");
        return NULL; 
    } 

    if (PyLong_Check(n_size_obj)) {
        n_size = (int)PyLong_AsLong(n_size_obj);

    } else if (PyFloat_Check(n_size_obj)) {
        double float_value = PyFloat_AsDouble(n_size_obj);
        n_size = (int)float_value;

    } else {
        PyErr_SetString(PyExc_TypeError, "build_array() expects an integer or a parsable float");
        return NULL;
    }  

    if (coordinate_system == NULL) {
        coordinate_system = "cartesian";
    }

    return build_ev_array(ev_sph, n_size, coordinate_system);
}


static PyObject *py_build_ev_ensemble(PyObject *self, PyObject *args) {
    EulerVector *ev_sph = (EulerVector *)self;
    PyObject *n_size_obj = NULL;
    int n_size;
    

    if (PyTuple_Size(args) != 1){
        PyErr_SetString(PyExc_TypeError, "build_ensemble() expects a single argument, the number of samples to draw from the covariance matrix");
        return NULL; 
    }

    if (!PyArg_ParseTuple(args, "O", &n_size_obj)) {
        PyErr_SetString(PyExc_TypeError, "Failed to parse input argument");
        return NULL;
    }

    if (PyLong_Check(n_size_obj)) {
        n_size = (int)PyLong_AsLong(n_size_obj);

    } else if (PyFloat_Check(n_size_obj)) {
        double float_value = PyFloat_AsDouble(n_size_obj);
        n_size = (int)float_value;

    } else {
        PyErr_SetString(PyExc_TypeError, "build_ensemble() expects an integer or a parsable float");
        return NULL;
    }  

    return build_ev_ensemble(ev_sph, n_size);
}




static PyObject *py_build_zero_array(PyObject *self, PyObject *args) {
    int n_size;
    
    if (!PyArg_ParseTuple(args, "i", &n_size)) {
        PyErr_SetString(PyExc_TypeError, "expected one argument, an integer with the number of samples");
        return NULL;
    }

    gsl_matrix *cA = gsl_matrix_calloc(3, n_size);
    return build_numpy_array(cA);
}
