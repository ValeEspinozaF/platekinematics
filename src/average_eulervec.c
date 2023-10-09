#include "average_ensemble.h"

static gsl_matrix* pylist_ev_to_gslmatrix(PyObject *py_array) {
    int n_size = (int)PyList_Size(py_array);
    gsl_matrix *ev_gsl = gsl_matrix_alloc(3, n_size);

    for (int i = 0; i < n_size; i++) {
        PyObject *ev_pyob = PyList_GetItem(py_array, i);
        /* if (!PyObject_TypeCheck(ev_pyob, &EulerVectorType)) {
            PySys_WriteStdout("Item %d is not an EulerVector instance\n", i);
        } */ //FIXME Not working whyyy

        EulerVector *ev = (EulerVector *)ev_pyob;
        double *ev_cart = sph2cart(ev->Lon, ev->Lat, ev->AngVelocity);

        gsl_matrix_set(ev_gsl, 0, i, ev_cart[0]);
        gsl_matrix_set(ev_gsl, 1, i, ev_cart[1]);
        gsl_matrix_set(ev_gsl, 2, i, ev_cart[2]);
    }

    return ev_gsl;
}

static PyObject *py_ev_average(PyObject *self, PyObject *args) {
    int n_args;
    PyObject *ev_pyob, *tr_pyobj = NULL;
    double *tr_list;

    PyObject *original_type, *original_value, *original_traceback; 

    
    n_args = (int)PyTuple_Size(args);
    if (n_args == 1 || n_args == 2) {
        if (!PyArg_ParseTuple(args, "O|O", &ev_pyob, &tr_pyobj)) {
            PyErr_SetString(PyExc_TypeError, "average() expects two arguments, a numpy array with the ensemble and tuple with the time range of the ensemble");
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "average() expects two arguments, a numpy array with the ensemble and tuple with the time range of the ensemble");
        return NULL; 
    } 


    // Time range arg parsing
    if (tr_pyobj != NULL) {
        tr_list = parse_double_array(tr_pyobj);
        if (tr_list == NULL) {
            PyErr_Fetch(&original_type, &original_value, &original_traceback);
            PyErr_Restore(original_type, original_value, original_traceback);
            return NULL;
        }

        if ((int)tr_list[0] != 2) {
            PyErr_SetString(PyExc_ValueError, "Time range argument must contain exactly 2 elements");
            return NULL;
        }

    } else {
        tr_list = (double *)malloc((2 + 1) * sizeof(double)); // default constructor
        tr_list[0] = 2.0;
        tr_list[1] = 0.0;
        tr_list[2] = 0.0;
    }

    // Ensemble arg parsing
    gsl_matrix* ev_gsl;
    if (PyArray_Check(ev_pyob)) {
        ev_gsl = pyarray2D_to_gslmatrix(ev_pyob);
        if (ev_gsl == NULL) {
            PyErr_Fetch(&original_type, &original_value, &original_traceback);
            PyErr_Restore(original_type, original_value, original_traceback);
            return NULL;
        }
        if (ev_gsl->size1 != 3) {
            PyErr_SetString(PyExc_TypeError, "Input ensemble array must have exactly three rows (x, y, z)");
            return NULL;
        }
    } else if (PyList_Check(ev_pyob)) {
        ev_gsl = pylist_ev_to_gslmatrix(ev_pyob);
        if (ev_gsl == NULL) {
            PyErr_Fetch(&original_type, &original_value, &original_traceback);
            PyErr_Restore(original_type, original_value, original_traceback);
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "average() expects a numpy array as input");
        return NULL;
    }
    

    // Calculate average vector from GSL matrix
    double* ev_cart = (double*)malloc(3 * sizeof(double));
    double* ev_cov = (double*)malloc(6 * sizeof(double));

    average_vector(ev_gsl, ev_cart, ev_cov);
    gsl_matrix_free(ev_gsl);

    double *ev_sph_vtr = cart2sph(ev_cart[0], ev_cart[1], ev_cart[2]);

    EulerVector *ev_avg = PyObject_New(EulerVector, &EulerVectorType);
    if (ev_avg != NULL) {
        ev_avg->Lon = ev_sph_vtr[0];
        ev_avg->Lat = ev_sph_vtr[1];
        ev_avg->AngVelocity = ev_sph_vtr[2];
        ev_avg->TimeRange[0] = tr_list[0 + 1];
        ev_avg->TimeRange[1] = tr_list[1 + 1];

        Covariance *cov = PyObject_New(Covariance, &CovarianceType);
        if (cov != NULL) {
            cov->C11 = ev_cov[0] * (M_PI / 180.0) * (M_PI / 180.0);
            cov->C12 = ev_cov[1] * (M_PI / 180.0) * (M_PI / 180.0);
            cov->C13 = ev_cov[2] * (M_PI / 180.0) * (M_PI / 180.0);
            cov->C22 = ev_cov[3] * (M_PI / 180.0) * (M_PI / 180.0);
            cov->C23 = ev_cov[4] * (M_PI / 180.0) * (M_PI / 180.0);
            cov->C33 = ev_cov[5] * (M_PI / 180.0) * (M_PI / 180.0);
            ev_avg->Covariance = *cov; 
            ev_avg->has_covariance = 1;
            
        } else {
            free(tr_list);
            PyErr_SetString(PyExc_RuntimeError, "Failed to create Covariance instance");
            return NULL;
        }

    } else {
        free(tr_list);
        PyErr_SetString(PyExc_RuntimeError, "Failed to create EulerVector instance");
        return NULL;
    }

    free(tr_list);
    return (PyObject *)ev_avg;
}
