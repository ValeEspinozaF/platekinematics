#include <numpy/arrayobject.h>

#include "../types/finrot.h"
#include "../parse_number.c"
#include "covariance_methods.c"
#include "../spherical_functions.c"
#include "../build_ensemble.c"

// Return a 3x3 rotation matrix expressed in radians.
gsl_matrix* to_rotation_matrix(const FiniteRot *fr_sph) {
    double x, y, z; 

    double *r = sph2cart(fr_sph->Lon, fr_sph->Lat, fr_sph->Angle);
    x = r[0];
    y = r[1];
    z = r[2];

    double a = to_radians(fr_sph->Angle);
    double b = 1.0 - cos(a);
    double c = sin(a);

    gsl_matrix *m = gsl_matrix_alloc(3, 3);

    // Rotation matrix in [radians]
    gsl_matrix_set(m, 0, 0, cos(a) + x * x * b);
    gsl_matrix_set(m, 0, 1, x * y * b - z * c);
    gsl_matrix_set(m, 0, 2, x * z * b + y * c);
    gsl_matrix_set(m, 1, 0, y * x * b + z * c);
    gsl_matrix_set(m, 1, 1, cos(a) + y * y * b);
    gsl_matrix_set(m, 1, 2, y * z * b - x * c);
    gsl_matrix_set(m, 2, 0, z * x * b - y * c);
    gsl_matrix_set(m, 2, 1, z * y * b + x * c);
    gsl_matrix_set(m, 2, 2, cos(a) + z * z * b);

    return m;
}


// Return the set of Euler angles from a finite rotation.
double* to_euler_angles(const FiniteRot *fr_sph) {
    gsl_matrix *matrix = to_rotation_matrix(fr_sph);
    double *euler_angles = (double*)malloc(3 * sizeof(double));

    euler_angles[0] = atan2(gsl_matrix_get(matrix, 2, 1), gsl_matrix_get(matrix, 2, 2));
    euler_angles[1] = atan2(-1.0 * gsl_matrix_get(matrix, 2, 0), sqrt(pow(gsl_matrix_get(matrix, 2, 1), 2) + pow(gsl_matrix_get(matrix, 2, 2), 2)));
    euler_angles[2] = atan2(gsl_matrix_get(matrix, 1, 0), gsl_matrix_get(matrix, 0, 0));

    return euler_angles;
}


// Draw n rotation matrix samples from the covariance of a given finite rotation.
gsl_matrix* build_fr_ensemble(FiniteRot *fr_sph, int n_size) {
    double *eu_angles;
    gsl_matrix *cov_matrix;
    gsl_matrix *correlated_ens;

    // ** If covariance is not given

    eu_angles = to_euler_angles(fr_sph);
    cov_matrix = fr_cov_to_matrix(fr_sph);
    

    // ** Take action if covariance-matrix has negative or imaginary eigenvalues

    correlated_ens = correlated_ensemble_3d(cov_matrix, n_size);

    for (size_t i = 0; i < 3; i++) {
        for(size_t j = 0; j < n_size; j++) {
            double current_value = gsl_matrix_get(correlated_ens, i, j);
            gsl_matrix_set(correlated_ens, i, j, current_value + eu_angles[i]);
        }
    }

    // ** ToRotationMatrix(EAx, EAy, EAz)
    return correlated_ens;
}

void capsule_cleanup(PyObject *capsule) {
    void *memory = PyCapsule_GetPointer(capsule, NULL);
    free(memory);
}

static PyObject *py_build_fr_ensemble(PyObject *self, PyObject *args) {
    FiniteRot *fr_sph = (FiniteRot*)self;
    PyObject *input_obj;
    gsl_matrix *correlated_ens;
    PyObject *np_array;
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

    correlated_ens = build_fr_ensemble(fr_sph, n_size);
    
    npy_intp dims[2];
    dims[0] = (int)correlated_ens->size1;
    dims[1] = (int)correlated_ens->size2;
    np_array = PyArray_SimpleNewFromData(2, dims, NPY_DOUBLE, (void *)correlated_ens->data);

    PyObject *capsule = PyCapsule_New(correlated_ens->data, NULL, capsule_cleanup);
    PyArray_SetBaseObject((PyArrayObject *) np_array, capsule);

    gsl_matrix_free(correlated_ens);
    return np_array;
}