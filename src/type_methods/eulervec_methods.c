#include "type_methods.h"


gsl_matrix * build_ev_array(EulerVector *ev_sph, int n_size, const char* coordinate_system) {
    gsl_matrix *cov_matrix, *correlated_ens;
    double * ev_cart, *_ev_sph;
    bool out_spherical;


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

    correlated_ens = correlated_ensemble_3d(cov_matrix, n_size);
    if (correlated_ens == NULL) {
        gsl_matrix_free(cov_matrix);
        PyErr_SetString(PyExc_MemoryError, "Failed to create correlated ensemble matrix");
        return NULL;
    }  

    gsl_matrix_free(cov_matrix);  

    for(size_t i = 0; i < n_size; i++) {
        double cx, cy, cz;
        cx = gsl_matrix_get(correlated_ens, 0, i) + ev_cart[0];
        cy = gsl_matrix_get(correlated_ens, 1, i) + ev_cart[1];
        cz = gsl_matrix_get(correlated_ens, 2, i) + ev_cart[2];
        if (out_spherical) {
            _ev_sph = cart2sph(cx, cy, cz);
            gsl_matrix_set(correlated_ens, 0, i, _ev_sph[0]);
            gsl_matrix_set(correlated_ens, 1, i, _ev_sph[1]);
            gsl_matrix_set(correlated_ens, 2, i, _ev_sph[2]);
        } else {
            gsl_matrix_set(correlated_ens, 0, i, cx);
            gsl_matrix_set(correlated_ens, 1, i, cy);
            gsl_matrix_set(correlated_ens, 2, i, cz);
        }
    }
    return correlated_ens;
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
        EulerVector *ev = PyObject_New(EulerVector, &EulerVectorType);
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


PyObject *py_build_ev_array(PyObject *self, PyObject *args) {
    EulerVector *ev_sph = (EulerVector *)self;
    PyObject *n_size_obj;
    int n_size, n_args;
    const char* coordinate_system = NULL;

    PyObject *original_type, *original_value, *original_traceback; 
    
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

    gsl_matrix* ev_array = build_ev_array(ev_sph, n_size, coordinate_system);
    if (ev_array == NULL) {
        PyErr_Fetch(&original_type, &original_value, &original_traceback);
        PyErr_Restore(original_type, original_value, original_traceback);
        return NULL;
    }

    return build_numpy_2Darray(ev_array);
}


PyObject *py_build_ev_ensemble(PyObject *self, PyObject *args) {
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