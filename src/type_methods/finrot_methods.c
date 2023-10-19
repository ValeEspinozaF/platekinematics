#include "type_methods.h"

gsl_vector* cov_to_gsl_vector(Covariance *cov);
PyObject* build_numpy_1Darray(gsl_vector *cA);
gsl_matrix* correlated_ensemble_3d(gsl_matrix *cov_matrix, int n_size);


gsl_vector* fr_to_gsl_vector(FiniteRot *fr_sph) {
    gsl_vector *fr_vector, *cov_vector, *fr_cov_vector;
    Covariance *cov;

    fr_vector = gsl_vector_alloc(4);
    gsl_vector_set(fr_vector, 0, fr_sph->Lon);
    gsl_vector_set(fr_vector, 1, fr_sph->Lat);
    gsl_vector_set(fr_vector, 2, fr_sph->Angle);
    gsl_vector_set(fr_vector, 3, fr_sph->Time);

    if (fr_sph->has_covariance == 1){
        cov = &(fr_sph->Covariance);
        cov_vector = cov_to_gsl_vector(cov);
        fr_cov_vector = gsl_vector_alloc(10);
        
        gsl_vector_view viewOut1 = gsl_vector_subvector(fr_cov_vector, 0, 4);
        gsl_vector_view viewOut2 = gsl_vector_subvector(fr_cov_vector, 4, 6);
        gsl_vector_memcpy(&viewOut1.vector, fr_vector);
        gsl_vector_memcpy(&viewOut2.vector, cov_vector);
        gsl_vector_free(fr_vector);
        gsl_vector_free(cov_vector);

    } else {
        fr_cov_vector = gsl_vector_alloc(4);
        
        gsl_vector_view viewOut1 = gsl_vector_subvector(fr_cov_vector, 0, 4);
        gsl_vector_memcpy(&viewOut1.vector, fr_vector);
        gsl_vector_free(fr_vector);
    }

    return fr_cov_vector;
}

PyObject *py_fr_to_numpy(PyObject *self, int Py_UNUSED(_)) {
    FiniteRot *fr_sph = (FiniteRot*)self;
    gsl_vector* fr_vector = fr_to_gsl_vector(fr_sph);
    return build_numpy_1Darray(fr_vector);
}


// Draw n_size rotation matrix samples from the covariance of a given finite rotation.
gsl_matrix** build_frm_array(FiniteRot *fr_sph, int n_size) {
    gsl_matrix *cov_matrix, *correlated_ens;
    double *eu_angles;

    PyObject *original_type, *original_value, *original_traceback; 

    if (fr_sph->has_covariance == 0){
        PyErr_SetString(PyExc_TypeError, "FiniteRotation must have a Covariance attribute");
        return NULL;
    }

    eu_angles = fr_to_euler_angles(fr_sph); // FIXME No NULL returns in to_euler_angles
    PySys_WriteStdout("eu_angles: %f, %f, %f\n", eu_angles[0], eu_angles[1], eu_angles[2]);

    if (eu_angles == NULL) {
        PyErr_Fetch(&original_type, &original_value, &original_traceback);
        PyErr_Restore(original_type, original_value, original_traceback);
        return NULL;
    }

    cov_matrix = fr_cov_to_matrix(fr_sph);
    PySys_WriteStdout("cov_matrix_row1: %f, %f, %f\n", gsl_matrix_get(cov_matrix, 0, 0), gsl_matrix_get(cov_matrix, 0, 1), gsl_matrix_get(cov_matrix, 0, 2));
    PySys_WriteStdout("cov_matrix_row2: %f, %f, %f\n", gsl_matrix_get(cov_matrix, 1, 0), gsl_matrix_get(cov_matrix, 1, 1), gsl_matrix_get(cov_matrix, 1, 2));
    PySys_WriteStdout("cov_matrix_row3: %f, %f, %f\n", gsl_matrix_get(cov_matrix, 2, 0), gsl_matrix_get(cov_matrix, 2, 1), gsl_matrix_get(cov_matrix, 2, 2));


    if (cov_matrix == NULL) {
        PyErr_Fetch(&original_type, &original_value, &original_traceback); // FIXME No NULL returns in fr_cov_to_matrix
        PyErr_Restore(original_type, original_value, original_traceback);
        return NULL;
    }

    // ** Take action if covariance-matrix has negative or imaginary eigenvalues

    correlated_ens = correlated_ensemble_3d(cov_matrix, n_size);
    if (correlated_ens == NULL) {
        gsl_matrix_free(cov_matrix);
        PyErr_SetString(PyExc_MemoryError, "Failed to create correlated ensemble matrix");
        return NULL;
    } 

    for (size_t i = 0; i < 3; i++) {
        for(size_t j = 0; j < n_size; j++) {
            double current_value = gsl_matrix_get(correlated_ens, i, j);
            gsl_matrix_set(correlated_ens, i, j, current_value + eu_angles[i]);
        }
    }

    gsl_matrix_free(cov_matrix);  

    gsl_vector *ex = gsl_vector_alloc(correlated_ens->size2);
    gsl_vector *ey = gsl_vector_alloc(correlated_ens->size2);
    gsl_vector *ez = gsl_vector_alloc(correlated_ens->size2);
    gsl_matrix_get_row(ex, correlated_ens, 0);
    gsl_matrix_get_row(ey, correlated_ens, 1);
    gsl_matrix_get_row(ez, correlated_ens, 2);

    return eas_to_rotation_matrices(ex, ey, ez);
}


PyObject *py_build_frm_array(PyObject *self, PyObject *args) {
    FiniteRot *fr_sph = (FiniteRot*)self;
    PyObject *n_size_obj;
    double float_value;
    int n_size;
    

    if (PyTuple_Size(args) != 1){
        PyErr_SetString(PyExc_TypeError, "build_array() expects a single argument, the number of samples to draw from the covariance matrix");
        return NULL; 
    }

    if (!PyArg_ParseTuple(args, "O", &n_size_obj)) {
        return NULL;
    }


    if (PyLong_Check(n_size_obj)) {
        PyArg_ParseTuple(args, "i", &n_size);
        
    } else if (PyFloat_Check(n_size_obj)) {
        PyArg_ParseTuple(args, "d", &float_value);
        n_size = (int)float_value;

    } else {
        PyErr_SetString(PyExc_TypeError, "build_array() expects an integer or a parsable float");
        return NULL;
    }    

    gsl_matrix** frm_array = build_frm_array(fr_sph, n_size);
    return build_numpy_3Darray(frm_array, n_size);
}