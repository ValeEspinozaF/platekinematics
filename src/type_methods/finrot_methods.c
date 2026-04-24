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


PyObject *build_fr_ensemble(FiniteRot *fr_sph, int n_size) {
    gsl_matrix **frm_array = build_frm_array(fr_sph, n_size);

    PyObject *original_type, *original_value, *original_traceback;
    PyErr_Fetch(&original_type, &original_value, &original_traceback);
    if (frm_array == NULL) {
        PyErr_Restore(original_type, original_value, original_traceback);
        return NULL;
    }

    PyObject *fr_ens = PyList_New(n_size);
    if (fr_ens == NULL) {
        for (int i = 0; i < n_size; ++i) {
            gsl_matrix_free(frm_array[i]);
        }
        free(frm_array);
        PyErr_SetString(PyExc_MemoryError, "Failed to create Python list");
        return NULL;
    }

    for (int i = 0; i < n_size; ++i) {
        FiniteRot *fr = rotation_matrix_to_fr(frm_array[i]);
        gsl_matrix_free(frm_array[i]);

        if (fr == NULL) {
            free(frm_array);
            Py_DECREF(fr_ens);
            PyErr_SetString(PyExc_RuntimeError, "Failed to create FiniteRotation instance");
            return NULL;
        }

        fr->Time = fr_sph->Time;
        fr->has_covariance = 0;
        PyList_SET_ITEM(fr_ens, i, (PyObject *)fr);
    }

    free(frm_array);
    return fr_ens;
}


// Draw n_size rotation matrix samples from the covariance of a given finite rotation.
gsl_matrix** build_frm_array(FiniteRot *fr_sph, int n_size) {
    PyObject *original_type, *original_value, *original_traceback; 
    gsl_matrix *cov_matrix, *correlated_ens;
    double *eu_angles;

    if (fr_sph->has_covariance == 0){
        PyErr_SetString(PyExc_TypeError, "FiniteRotation must have a Covariance attribute");
        return NULL;
    }

    eu_angles = fr_to_euler_angles(fr_sph); // FIXME No NULL returns in to_euler_angles
    if (eu_angles == NULL) {
        PyErr_Fetch(&original_type, &original_value, &original_traceback);
        PyErr_Restore(original_type, original_value, original_traceback);
        return NULL;
    }

    cov_matrix = fr_cov_to_matrix(fr_sph);
    if (cov_matrix == NULL) {
        PyErr_Fetch(&original_type, &original_value, &original_traceback); // FIXME No NULL returns in fr_cov_to_matrix
        PyErr_Restore(original_type, original_value, original_traceback);
        return NULL;
    }


    correlated_ens = correlated_ensemble_3d(cov_matrix, n_size);
    if (correlated_ens == NULL) {
        gsl_matrix_free(cov_matrix);
        PyErr_Fetch(&original_type, &original_value, &original_traceback);
        PyErr_Restore(original_type, original_value, original_traceback);
        return NULL;
    }  
    gsl_matrix_free(cov_matrix);  


    for (size_t i = 0; i < 3; i++) {
        for(size_t j = 0; j < n_size; j++) {
            double current_value = gsl_matrix_get(correlated_ens, i, j);
            gsl_matrix_set(correlated_ens, i, j, current_value + eu_angles[i]);
        }
    }
    

    gsl_vector *ex = gsl_vector_alloc(correlated_ens->size2);
    gsl_vector *ey = gsl_vector_alloc(correlated_ens->size2);
    gsl_vector *ez = gsl_vector_alloc(correlated_ens->size2);
    gsl_matrix_get_row(ex, correlated_ens, 0);
    gsl_matrix_get_row(ey, correlated_ens, 1);
    gsl_matrix_get_row(ez, correlated_ens, 2);
    gsl_matrix_free(correlated_ens);

    return eas_to_rotation_matrices(ex, ey, ez);
}


PyObject *py_build_frm_array(PyObject *self, PyObject *args) {
    PyObject *original_type, *original_value, *original_traceback; 
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
    if (frm_array == NULL) {
        PyErr_Fetch(&original_type, &original_value, &original_traceback);
        PyErr_Restore(original_type, original_value, original_traceback);
        return NULL;
    }  

    return build_numpy_3Darray(frm_array, n_size);
}


PyObject *py_build_fr_ensemble(PyObject *self, PyObject *args) {
    FiniteRot *fr_sph = (FiniteRot *)self;
    PyObject *n_size_obj = NULL;
    int n_size;

    if (PyTuple_Size(args) != 1) {
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

    return build_fr_ensemble(fr_sph, n_size);
}