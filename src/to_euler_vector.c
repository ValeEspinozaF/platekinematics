#define PY_ARRAY_UNIQUE_SYMBOL PLATEKIN_ARRAY_API
#define NO_IMPORT_ARRAY
#include "pk_structs.h"
#include "spherical_functions.h"
#include "type_conversions/type_conversions.h"

PyObject *py_ev_average(PyObject *self, PyObject *args);
gsl_matrix **build_frm_array(FiniteRot *fr_sph, int n_size);

static int covariance_is_zero(const Covariance *cov) {
    return cov->C11 == 0.0 && cov->C12 == 0.0 && cov->C13 == 0.0 &&
           cov->C22 == 0.0 && cov->C23 == 0.0 && cov->C33 == 0.0;
}

static int fr_has_nonzero_covariance(const FiniteRot *fr) {
    return fr->has_covariance == 1 && !covariance_is_zero(&fr->Covariance);
}

static gsl_matrix *matrix_transpose_copy(const gsl_matrix *m) {
    gsl_matrix *out = gsl_matrix_alloc(3, 3);
    if (out == NULL) {
        return NULL;
    }

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            gsl_matrix_set(out, i, j, gsl_matrix_get(m, j, i));
        }
    }
    return out;
}

static gsl_matrix *stage_matrix_from_pair(const gsl_matrix *m1, const gsl_matrix *m2) {
    gsl_matrix *inv1 = matrix_transpose_copy(m1);
    gsl_matrix *out;

    if (inv1 == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    out = gsl_matrix_alloc(3, 3);
    if (out == NULL) {
        gsl_matrix_free(inv1);
        PyErr_NoMemory();
        return NULL;
    }

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
            double sum = 0.0;
            for (size_t k = 0; k < 3; k++) {
                sum += gsl_matrix_get(m2, i, k) * gsl_matrix_get(inv1, k, j);
            }
            gsl_matrix_set(out, i, j, sum);
        }
    }

    gsl_matrix_free(inv1);
    return out;
}

static EulerVector *euler_vector_from_matrix(const gsl_matrix *m, double t0, double t1) {
    double x = gsl_matrix_get(m, 2, 1) - gsl_matrix_get(m, 1, 2);
    double y = gsl_matrix_get(m, 0, 2) - gsl_matrix_get(m, 2, 0);
    double z = gsl_matrix_get(m, 1, 0) - gsl_matrix_get(m, 0, 1);

    double sph[3];
    double trace;
    double dtime;
    double vel;

    cart2sph(x, y, z, sph);
    trace = gsl_matrix_get(m, 0, 0) + gsl_matrix_get(m, 1, 1) + gsl_matrix_get(m, 2, 2);
    dtime = fabs(t1 - t0);

    if (dtime == 0.0) {
        PyErr_SetString(PyExc_ValueError, "Cannot convert to EulerVector with zero time span");
        return NULL;
    }

    vel = to_degrees(atan2(sph[2], trace - 1.0)) / dtime;

    EulerVector *ev = PyObject_New(EulerVector, &EulerVectorType);
    if (ev == NULL) {
        PyErr_NoMemory();
        return NULL;
    }

    ev->Lon = sph[0];
    ev->Lat = sph[1];
    ev->AngVelocity = vel;
    ev->TimeRange[0] = t0;
    ev->TimeRange[1] = t1;
    ev->has_covariance = 0;
    ev->Covariance.C11 = 0.0;
    ev->Covariance.C12 = 0.0;
    ev->Covariance.C13 = 0.0;
    ev->Covariance.C22 = 0.0;
    ev->Covariance.C23 = 0.0;
    ev->Covariance.C33 = 0.0;

    return ev;
}

static PyObject *average_euler_list(PyObject *ev_list, double t0, double t1) {
    PyObject *time_range = Py_BuildValue("(dd)", t0, t1);
    PyObject *args;
    PyObject *result;

    if (time_range == NULL) {
        return NULL;
    }

    args = PyTuple_Pack(2, ev_list, time_range);
    Py_DECREF(time_range);
    if (args == NULL) {
        return NULL;
    }

    result = py_ev_average(NULL, args);
    Py_DECREF(args);
    return result;
}

static int parse_optional_double(PyObject *obj, const char *name, double *out, int *is_set) {
    if (obj == NULL || obj == Py_None) {
        *is_set = 0;
        return 0;
    }

    if (!(PyFloat_Check(obj) || PyLong_Check(obj))) {
        PyErr_Format(PyExc_TypeError, "%s must be a float when provided", name);
        return -1;
    }

    *out = PyFloat_AsDouble(obj);
    *is_set = 1;
    return 0;
}

static int extract_fr_list(PyObject *list_obj, FiniteRot ***out_list, int *out_n) {
    int n;
    FiniteRot **items;

    if (!PyList_Check(list_obj)) {
        PyErr_SetString(PyExc_TypeError, "Expected a list of FiniteRotation objects");
        return -1;
    }

    n = (int)PyList_Size(list_obj);
    if (n <= 0) {
        PyErr_SetString(PyExc_ValueError, "FiniteRotation list cannot be empty");
        return -1;
    }

    items = (FiniteRot **)malloc((size_t)n * sizeof(FiniteRot *));
    if (items == NULL) {
        PyErr_NoMemory();
        return -1;
    }

    for (int i = 0; i < n; i++) {
        PyObject *item = PyList_GetItem(list_obj, i);
        if (!PyObject_TypeCheck(item, &FiniteRotationType)) {
            free(items);
            PyErr_Format(PyExc_TypeError, "Item %d is not a FiniteRotation", i);
            return -1;
        }
        items[i] = (FiniteRot *)item;
    }

    *out_list = items;
    *out_n = n;
    return 0;
}

static PyObject *to_euler_vector_single(FiniteRot *fr, int reverse_rot, int n_size) {
    double t0 = reverse_rot ? 0.0 : fr->Time;
    double t1 = reverse_rot ? fr->Time : 0.0;
    int use_ensemble = fr_has_nonzero_covariance(fr);

    if (use_ensemble) {
        gsl_matrix **arr = build_frm_array(fr, n_size);
        PyObject *ev_list;

        if (arr == NULL) {
            return NULL;
        }

        ev_list = PyList_New(n_size);
        if (ev_list == NULL) {
            for (int i = 0; i < n_size; i++) {
                gsl_matrix_free(arr[i]);
            }
            free(arr);
            return NULL;
        }

        for (int i = 0; i < n_size; i++) {
            gsl_matrix *m = arr[i];
            gsl_matrix *active = reverse_rot ? m : matrix_transpose_copy(m);
            EulerVector *ev;

            if (!reverse_rot && active == NULL) {
                Py_DECREF(ev_list);
                for (int j = i; j < n_size; j++) {
                    gsl_matrix_free(arr[j]);
                }
                for (int j = 0; j < i; j++) {
                    ;
                }
                free(arr);
                PyErr_NoMemory();
                return NULL;
            }

            ev = euler_vector_from_matrix(active, t0, t1);
            if (!reverse_rot) {
                gsl_matrix_free(active);
            }
            gsl_matrix_free(m);

            if (ev == NULL) {
                Py_DECREF(ev_list);
                for (int j = i + 1; j < n_size; j++) {
                    gsl_matrix_free(arr[j]);
                }
                free(arr);
                return NULL;
            }

            PyList_SET_ITEM(ev_list, i, (PyObject *)ev);
        }

        free(arr);
        return average_euler_list(ev_list, t0, t1);
    }

    gsl_matrix *m = fr_to_rotation_matrix(fr);
    gsl_matrix *active = reverse_rot ? m : matrix_transpose_copy(m);
    EulerVector *ev;

    if (m == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to build rotation matrix from FiniteRotation");
        return NULL;
    }

    if (!reverse_rot && active == NULL) {
        gsl_matrix_free(m);
        PyErr_NoMemory();
        return NULL;
    }

    ev = euler_vector_from_matrix(active, t0, t1);
    if (!reverse_rot) {
        gsl_matrix_free(active);
    }
    gsl_matrix_free(m);
    return (PyObject *)ev;
}

static PyObject *to_euler_vector_pair(FiniteRot *fr1, FiniteRot *fr2, int reverse_rot, int n_size) {
    int use_ensemble = fr_has_nonzero_covariance(fr1) && fr_has_nonzero_covariance(fr2);
    double t0 = reverse_rot ? fr2->Time : fr1->Time;
    double t1 = reverse_rot ? fr1->Time : fr2->Time;

    if (use_ensemble) {
        gsl_matrix **arr1 = build_frm_array(fr1, n_size);
        gsl_matrix **arr2 = NULL;
        PyObject *ev_list = NULL;

        if (arr1 == NULL) {
            return NULL;
        }

        arr2 = build_frm_array(fr2, n_size);
        if (arr2 == NULL) {
            for (int i = 0; i < n_size; i++) {
                gsl_matrix_free(arr1[i]);
            }
            free(arr1);
            return NULL;
        }

        ev_list = PyList_New(n_size);
        if (ev_list == NULL) {
            for (int i = 0; i < n_size; i++) {
                gsl_matrix_free(arr1[i]);
                gsl_matrix_free(arr2[i]);
            }
            free(arr1);
            free(arr2);
            return NULL;
        }

        for (int i = 0; i < n_size; i++) {
            gsl_matrix *left = reverse_rot ? arr2[i] : arr1[i];
            gsl_matrix *right = reverse_rot ? arr1[i] : arr2[i];
            gsl_matrix *stage = stage_matrix_from_pair(left, right);
            EulerVector *ev;

            gsl_matrix_free(arr1[i]);
            gsl_matrix_free(arr2[i]);

            if (stage == NULL) {
                Py_DECREF(ev_list);
                for (int j = i + 1; j < n_size; j++) {
                    gsl_matrix_free(arr1[j]);
                    gsl_matrix_free(arr2[j]);
                }
                free(arr1);
                free(arr2);
                return NULL;
            }

            ev = euler_vector_from_matrix(stage, t0, t1);
            gsl_matrix_free(stage);
            if (ev == NULL) {
                Py_DECREF(ev_list);
                for (int j = i + 1; j < n_size; j++) {
                    gsl_matrix_free(arr1[j]);
                    gsl_matrix_free(arr2[j]);
                }
                free(arr1);
                free(arr2);
                return NULL;
            }

            PyList_SET_ITEM(ev_list, i, (PyObject *)ev);
        }

        free(arr1);
        free(arr2);
        return average_euler_list(ev_list, t0, t1);
    }

    gsl_matrix *m1 = fr_to_rotation_matrix(fr1);
    gsl_matrix *m2 = fr_to_rotation_matrix(fr2);
    gsl_matrix *stage;
    EulerVector *ev;

    if (m1 == NULL || m2 == NULL) {
        if (m1 != NULL) {
            gsl_matrix_free(m1);
        }
        if (m2 != NULL) {
            gsl_matrix_free(m2);
        }
        PyErr_SetString(PyExc_RuntimeError, "Failed to build rotation matrix from FiniteRotation pair");
        return NULL;
    }

    stage = reverse_rot ? stage_matrix_from_pair(m2, m1) : stage_matrix_from_pair(m1, m2);
    gsl_matrix_free(m1);
    gsl_matrix_free(m2);

    if (stage == NULL) {
        return NULL;
    }

    ev = euler_vector_from_matrix(stage, t0, t1);
    gsl_matrix_free(stage);
    return (PyObject *)ev;
}

static PyObject *to_euler_vector_samples(PyObject *fr_list_obj, PyObject *time_obj, int reverse_rot) {
    FiniteRot **frs = NULL;
    int n = 0;
    double time = 0.0;
    int has_time = 0;
    double t0;
    double t1;
    PyObject *ev_list;

    if (extract_fr_list(fr_list_obj, &frs, &n) != 0) {
        return NULL;
    }

    if (n > 1 && frs[0]->Time != frs[1]->Time) {
        free(frs);
        PyErr_SetString(PyExc_ValueError,
                        "This list overload is for sampled FiniteRotation arrays of a single time");
        return NULL;
    }

    if (parse_optional_double(time_obj, "time", &time, &has_time) != 0) {
        free(frs);
        return NULL;
    }

    if (!has_time) {
        time = frs[0]->Time;
    }

    t0 = reverse_rot ? 0.0 : time;
    t1 = reverse_rot ? time : 0.0;

    ev_list = PyList_New(n);
    if (ev_list == NULL) {
        free(frs);
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        gsl_matrix *m = fr_to_rotation_matrix(frs[i]);
        gsl_matrix *active = reverse_rot ? m : matrix_transpose_copy(m);
        EulerVector *ev;

        if (m == NULL || (!reverse_rot && active == NULL)) {
            if (m != NULL) {
                gsl_matrix_free(m);
            }
            if (!reverse_rot && active != NULL) {
                gsl_matrix_free(active);
            }
            Py_DECREF(ev_list);
            free(frs);
            PyErr_SetString(PyExc_RuntimeError, "Failed to build rotation matrix from sampled FiniteRotation");
            return NULL;
        }

        ev = euler_vector_from_matrix(active, t0, t1);
        if (!reverse_rot) {
            gsl_matrix_free(active);
        }
        gsl_matrix_free(m);

        if (ev == NULL) {
            Py_DECREF(ev_list);
            free(frs);
            return NULL;
        }

        PyList_SET_ITEM(ev_list, i, (PyObject *)ev);
    }

    free(frs);

    if (n == 1) {
        PyObject *only = PyList_GetItem(ev_list, 0);
        Py_INCREF(only);
        Py_DECREF(ev_list);
        return only;
    }

    return average_euler_list(ev_list, t0, t1);
}

static PyObject *to_euler_vector_sample_pairs(PyObject *fr1_list_obj,
                                               PyObject *fr2_list_obj,
                                               PyObject *time1_obj,
                                               PyObject *time2_obj,
                                               int reverse_rot) {
    FiniteRot **fr1 = NULL;
    FiniteRot **fr2 = NULL;
    int n1 = 0;
    int n2 = 0;
    double time1 = 0.0;
    double time2 = 0.0;
    int has_time1 = 0;
    int has_time2 = 0;
    double t0;
    double t1;
    PyObject *ev_list;

    if (extract_fr_list(fr1_list_obj, &fr1, &n1) != 0) {
        return NULL;
    }
    if (extract_fr_list(fr2_list_obj, &fr2, &n2) != 0) {
        free(fr1);
        return NULL;
    }

    if (n1 != n2) {
        free(fr1);
        free(fr2);
        PyErr_SetString(PyExc_ValueError, "Input FiniteRotation sample lists must have same length");
        return NULL;
    }

    if (parse_optional_double(time1_obj, "time1", &time1, &has_time1) != 0 ||
        parse_optional_double(time2_obj, "time2", &time2, &has_time2) != 0) {
        free(fr1);
        free(fr2);
        return NULL;
    }

    if (!has_time1) {
        time1 = fr1[0]->Time;
    }
    if (!has_time2) {
        time2 = fr2[0]->Time;
    }

    t0 = reverse_rot ? time2 : time1;
    t1 = reverse_rot ? time1 : time2;

    ev_list = PyList_New(n1);
    if (ev_list == NULL) {
        free(fr1);
        free(fr2);
        return NULL;
    }

    for (int i = 0; i < n1; i++) {
        gsl_matrix *m1 = fr_to_rotation_matrix(fr1[i]);
        gsl_matrix *m2 = fr_to_rotation_matrix(fr2[i]);
        gsl_matrix *stage;
        EulerVector *ev;

        if (m1 == NULL || m2 == NULL) {
            if (m1 != NULL) {
                gsl_matrix_free(m1);
            }
            if (m2 != NULL) {
                gsl_matrix_free(m2);
            }
            Py_DECREF(ev_list);
            free(fr1);
            free(fr2);
            PyErr_SetString(PyExc_RuntimeError, "Failed to build rotation matrix from sampled pair");
            return NULL;
        }

        stage = reverse_rot ? stage_matrix_from_pair(m2, m1) : stage_matrix_from_pair(m1, m2);
        gsl_matrix_free(m1);
        gsl_matrix_free(m2);

        if (stage == NULL) {
            Py_DECREF(ev_list);
            free(fr1);
            free(fr2);
            return NULL;
        }

        ev = euler_vector_from_matrix(stage, t0, t1);
        gsl_matrix_free(stage);

        if (ev == NULL) {
            Py_DECREF(ev_list);
            free(fr1);
            free(fr2);
            return NULL;
        }

        PyList_SET_ITEM(ev_list, i, (PyObject *)ev);
    }

    free(fr1);
    free(fr2);

    if (n1 == 1) {
        PyObject *only = PyList_GetItem(ev_list, 0);
        Py_INCREF(only);
        Py_DECREF(ev_list);
        return only;
    }

    return average_euler_list(ev_list, t0, t1);
}

PyObject *py_to_euler_vector(PyObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"obj1", "obj2", "reverse_rot", "n_size", "time", "time1", "time2", NULL};

    PyObject *obj1;
    PyObject *obj2 = Py_None;
    int reverse_rot = 0;
    int n_size = 100000;
    PyObject *time_obj = Py_None;
    PyObject *time1_obj = Py_None;
    PyObject *time2_obj = Py_None;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OpiOOO", kwlist,
                                     &obj1, &obj2, &reverse_rot, &n_size,
                                     &time_obj, &time1_obj, &time2_obj)) {
        return NULL;
    }

    if (n_size <= 0) {
        PyErr_SetString(PyExc_ValueError, "n_size must be a positive integer");
        return NULL;
    }

    if (PyObject_TypeCheck(obj1, &FiniteRotationType)) {
        if (obj2 == Py_None) {
            return to_euler_vector_single((FiniteRot *)obj1, reverse_rot, n_size);
        }
        if (PyObject_TypeCheck(obj2, &FiniteRotationType)) {
            return to_euler_vector_pair((FiniteRot *)obj1, (FiniteRot *)obj2, reverse_rot, n_size);
        }

        PyErr_SetString(PyExc_TypeError,
                        "When obj1 is FiniteRotation, obj2 must be FiniteRotation or omitted");
        return NULL;
    }

    if (PyList_Check(obj1)) {
        if (obj2 == Py_None) {
            return to_euler_vector_samples(obj1, time_obj, reverse_rot);
        }
        if (PyList_Check(obj2)) {
            return to_euler_vector_sample_pairs(obj1, obj2, time1_obj, time2_obj, reverse_rot);
        }

        PyErr_SetString(PyExc_TypeError,
                        "When obj1 is a list, obj2 must be a list or omitted");
        return NULL;
    }

    PyErr_SetString(PyExc_TypeError,
                    "to_euler_vector expects FiniteRotation or list of FiniteRotation as first argument");
    return NULL;
}

PyObject *py_to_euler_vector_list(PyObject *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"fr_list", "reverse_rot", "n_size", NULL};

    PyObject *fr_list_obj;
    int reverse_rot = 0;
    int n_size = 100000;
    FiniteRot **frs = NULL;
    int n = 0;
    PyObject *out;

    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|pi", kwlist,
                                     &fr_list_obj, &reverse_rot, &n_size)) {
        return NULL;
    }

    if (n_size <= 0) {
        PyErr_SetString(PyExc_ValueError, "n_size must be a positive integer");
        return NULL;
    }

    if (extract_fr_list(fr_list_obj, &frs, &n) != 0) {
        return NULL;
    }

    out = PyList_New(n);
    if (out == NULL) {
        free(frs);
        return NULL;
    }

    for (int i = 0; i < n; i++) {
        PyObject *ev;

        if (i == 0) {
            ev = to_euler_vector_single(frs[i], !reverse_rot, n_size);
        } else {
            ev = to_euler_vector_pair(frs[i - 1], frs[i], reverse_rot, n_size);
        }

        if (ev == NULL) {
            Py_DECREF(out);
            free(frs);
            return NULL;
        }

        PyList_SET_ITEM(out, i, ev);
    }

    free(frs);
    return out;
}
