#include <Python.h>
#include "structmember.h"
#include "../type_methods/eulervec_methods.c"
//#include "../parse_array.c"
//#include "../types/covariance.c"

static PyMemberDef EulerVector_members[] = {
    {"Lon", T_DOUBLE, offsetof(EulerVector, Lon), 0, "Longitude of the Euler pole in degrees-East."},
    {"Lat", T_DOUBLE, offsetof(EulerVector, Lat), 0, "Latitude of the Euler pole in degrees-North."},
    {"AngVelocity", T_DOUBLE, offsetof(EulerVector, AngVelocity), 0, "Angular velocity in degrees/Myr."},
    //{"TimeRange", T_DOUBLE | T_2BYTES, offsetof(EulerVector, TimeRange), READONLY, "Initial to final age of rotation."},
    //{"Covariance", T_OBJECT_EX, offsetof(EulerVector, Covariance), READONLY, "Covariance in radians²/Myr²."},
    {NULL} // Sentinel
};


static PyObject* EulerVector_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyObject *lon_obj, *lat_obj, *angvel_obj;
    PyObject *time_range_obj = NULL, *cov = NULL;
    
    int has_covariance;
    double lon, lat, angvel, tr1, tr2;
    

    if (kwds != NULL && PyDict_Size(kwds) > 0) {
        PyErr_SetString(PyExc_TypeError, "EulerVector() does not accept keyword arguments");
        return NULL;
    }

    if (PyTuple_Size(args) == 0 || args == NULL){ // Default constructor
        lon = 0.0;
        lat = 0.0;
        angvel = 0.0;
        tr1 = 0.0;
        tr2 = 0.0;
        has_covariance = 0;

    } else if (PyTuple_Size(args) == 5 || PyTuple_Size(args) == 4) {

        if (!PyArg_ParseTuple(args, "OOOO|O", &lon_obj, &lat_obj, &angvel_obj, &time_range_obj, &cov)) {
            Py_XDECREF(lon_obj);
            Py_XDECREF(lat_obj);
            Py_XDECREF(angvel_obj);
            Py_XDECREF(time_range_obj);
            Py_XDECREF(cov);
            PyErr_SetString(PyExc_TypeError, "EulerVector() failed to parse one or more input arguments");
            return NULL;
        }

        if (!PyTuple_Check(time_range_obj) || PyTuple_Size(time_range_obj) != 2) {
            Py_XDECREF(lon_obj);
            Py_XDECREF(lat_obj);
            Py_XDECREF(angvel_obj);
            Py_XDECREF(time_range_obj);
            Py_XDECREF(cov);
            PyErr_SetString(PyExc_TypeError, "TimeRange must be a tuple of two elements");
            return NULL;
        }

        PyObject* value1 = PyTuple_GetItem(time_range_obj, 0);
        PyObject* value2 = PyTuple_GetItem(time_range_obj, 1);
        Py_XDECREF(time_range_obj);

        if (!PyFloat_Check(value1) || !PyFloat_Check(value2)) {
            Py_XDECREF(lon_obj);
            Py_XDECREF(lat_obj);
            Py_XDECREF(angvel_obj);
            Py_XDECREF(cov);
            Py_XDECREF(value1);
            Py_XDECREF(value2);
            PyErr_SetString(PyExc_TypeError, "TimeRange elements must be of type float");
            return NULL;
        }

        tr1 = PyFloat_AsDouble(value1);
        tr2 = PyFloat_AsDouble(value2);
        Py_XDECREF(value1);
        Py_XDECREF(value2);


        // Check if Lon, Lat, AngVelocity arguments are doubles
        if (!PyFloat_Check(lon_obj) || !PyFloat_Check(lat_obj) || !PyFloat_Check(angvel_obj)) {
            Py_XDECREF(lon_obj);
            Py_XDECREF(lat_obj);
            Py_XDECREF(angvel_obj);
            Py_XDECREF(cov);
            PyErr_SetString(PyExc_TypeError, "Lon, Lat and AngVelocity arguments must be doubles");
            return NULL;

        } else {
            lon = PyFloat_AsDouble(lon_obj);
            lat = PyFloat_AsDouble(lat_obj);
            angvel = PyFloat_AsDouble(angvel_obj);
            Py_XDECREF(lon_obj);
            Py_XDECREF(lat_obj);
            Py_XDECREF(angvel_obj);
        }

        if (cov == NULL) {
            has_covariance = 0;
        } else {
            if (PyObject_IsInstance(cov, (PyObject *)&CovarianceType)) {
                has_covariance = 1;
            } else {
                Py_XDECREF(cov);
                PyErr_SetString(PyExc_TypeError, "Covariance argument must be of type Covariance()");
                return NULL;
            }
        }
    }
    else {
        PyErr_SetString(PyExc_TypeError, "EulerVector() constructor accepts at four or five arguments: Lon, Lat, AngVelocity, TimeRange, *Covariance");
        return NULL;
    }

    EulerVector *self = (EulerVector *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->Lon = lon;
        self->Lat = lat;
        self->AngVelocity = angvel;
        self->TimeRange[0] = tr1;
        self->TimeRange[1] = tr2;
        self->has_covariance = has_covariance;

        if (has_covariance) {
            self->Covariance = *((Covariance *)cov);
            //Py_XDECREF(cov);
        }
    }
    return (PyObject *)self;
}


// Representation of EulerVector object
static PyObject* EulerVector_repr(EulerVector *self) {
    const char *format = "EulerVector(Lon=%.2f, Lat=%.2f, AngVelocity=%.3f, TimeRange=[%.2f, %.2f])";
    char buffer[256];  // Adjust the buffer size as needed

    snprintf(buffer, sizeof(buffer), format,
             self->Lon, self->Lat, self->AngVelocity, self->TimeRange[0], self->TimeRange[1]);

    return PyUnicode_FromString(buffer);
}


// Destructor function for EulerVector
static void EulerVector_dealloc(EulerVector *self) {
    Covariance_dealloc(&(self->Covariance));
    Py_TYPE(self)->tp_free(self);
}


// Getter functions for FEulerVector attributes
static PyObject* EulerVector_get_Lon(EulerVector *self, void *closure) {
    return PyFloat_FromDouble(self->Lon);
}

static PyObject* EulerVector_get_Lat(EulerVector *self, void *closure) {
    return PyFloat_FromDouble(self->Lat);
}

static PyObject* EulerVector_get_AngVelocity(EulerVector *self, void *closure) {
    return PyFloat_FromDouble(self->AngVelocity);
}

static PyObject* EulerVector_get_TimeRange(EulerVector *self, void *closure) {
    PyObject* time_range_tuple = PyTuple_New(2);

    if (time_range_tuple) {
        PyTuple_SET_ITEM(time_range_tuple, 0, PyFloat_FromDouble(self->TimeRange[0]));
        PyTuple_SET_ITEM(time_range_tuple, 1, PyFloat_FromDouble(self->TimeRange[1]));
    }

    return time_range_tuple;
}

static PyObject* EulerVector_get_Covariance(EulerVector *self, void *closure) {
    if (!self->has_covariance) {
        PySys_WriteStdout("No covariance assigned\n");
        Py_RETURN_NONE;
    }

    Covariance *cov = &(self->Covariance);
    if (cov == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to get Covariance attribute");
        Py_RETURN_NONE;
    } else {
        Py_INCREF(cov);
        return (PyObject *)cov;
    }
} 


// Setter functions for FiniteRotation attributes
static int EulerVector_set_Lon(EulerVector *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
        return -1;
    }
    if (!PyFloat_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Lon attribute must be a float");
        return -1;
    }
    self->Lon = PyFloat_AsDouble(value);
    return 0;
}

static int EulerVector_set_Lat(EulerVector *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
        return -1;
    }
    if (!PyFloat_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Lat attribute must be a float");
        return -1;
    }
    self->Lat = PyFloat_AsDouble(value);
    return 0;
}

static int EulerVector_set_AngVelocity(EulerVector *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
        return -1;
    }
    if (!PyFloat_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "AngVelocity attribute must be a float");
        return -1;
    }
    self->AngVelocity = PyFloat_AsDouble(value);
    return 0;
}

static int EulerVector_set_TimeRange(EulerVector *self, PyObject *list, void *closure) {
    if (!PyTuple_Check(list) || PyTuple_Size(list) != 2) {
        PyErr_SetString(PyExc_TypeError, "TimeRange must be a tuple of two elements");
        return -1;
    }

    PyObject* value1 = PyTuple_GetItem(list, 0);
    PyObject* value2 = PyTuple_GetItem(list, 1);

    if (!PyFloat_Check(value1) || !PyFloat_Check(value2)) {
        PyErr_SetString(PyExc_TypeError, "TimeRange elements must be of type float");
        return -1;
    }

    self->TimeRange[0] = PyFloat_AsDouble(value1);
    self->TimeRange[1] = PyFloat_AsDouble(value2);

    return 0;
}

static int EulerVector_set_Covariance(EulerVector *self, PyObject *cov_instance, void *closure) {
    if (!PyObject_TypeCheck(cov_instance, &CovarianceType)) {
        PyErr_SetString(PyExc_TypeError, "Covariance attribute must be a Covariance instance");
        return -1;
    }

    Covariance *cov = &(self->Covariance);
    Py_XDECREF((PyObject *)cov);
    Py_INCREF(cov_instance);
    self->Covariance = *((Covariance *)cov_instance);
    self->has_covariance = 1;
    return 0;
} 

static PyGetSetDef EulerVector_getsetters[] = {
    {"Lon", (getter)EulerVector_get_Lon, (setter)EulerVector_set_Lon, "Longitude", NULL},
    {"Lat", (getter)EulerVector_get_Lat, (setter)EulerVector_set_Lat, "Latitude", NULL},
    {"AngVelocity", (getter)EulerVector_get_AngVelocity, (setter)EulerVector_set_AngVelocity, "AngVelocity", NULL},
    {"TimeRange", (getter)EulerVector_get_TimeRange, (setter)EulerVector_set_TimeRange, "TimeRange", NULL},
    {"Covariance", (getter)EulerVector_get_Covariance, (setter)EulerVector_set_Covariance, "Covariance attribute", NULL},
    {NULL}
};




static PyMethodDef EulerVector_methods[] = {
    {"build_ensemble", py_build_ev_ensemble, METH_VARARGS, "Draws n EulerVector() samples from the covariance of a given Euler vector."},
    {"build_array", py_build_ev_array, METH_VARARGS, "Draws n Euler vector coordinate samples and stores them in a 3byn array."},
    {"build_numpy", py_build_zero_array, METH_VARARGS, "Build a numpy array of 3 by n_size, all elements initialized to zero."},
    {NULL, NULL, 0, NULL}
}; 


static PyTypeObject EulerVectorType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pk_structs.EulerVector",
    .tp_doc = "EulerVector object",
    .tp_members = EulerVector_members,
    .tp_repr = (reprfunc)EulerVector_repr,
    .tp_basicsize = sizeof(EulerVector),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = EulerVector_new,
    .tp_dealloc = (destructor)EulerVector_dealloc,
    .tp_getset = EulerVector_getsetters,
    .tp_methods = EulerVector_methods,
};