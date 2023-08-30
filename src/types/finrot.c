#include <Python.h>
#include "structmember.h"
#include "../type_methods/finrot_methods.c"
//#include "../types/covariance.c"

static PyMemberDef FiniteRotation_members[] = {
    {"Lon", T_DOUBLE, offsetof(FiniteRot, Lon), 0, "Longitude of the rotation axis in degrees-East."},
    {"Lat", T_DOUBLE, offsetof(FiniteRot, Lat), 0, "Latitude of the rotation axis in degrees-North."},
    {"Angle", T_DOUBLE, offsetof(FiniteRot, Angle), 0, "Angle of rotation in degrees."},
    {"Time", T_DOUBLE, offsetof(FiniteRot, Time), 0, "Age of rotation in million years."},
    //dont even try, you will crash something
    //{"Covariance", T_OBJECT_EX, offsetof(FiniteRot, Covariance), READONLY, "Covariance in radians2."},
    {NULL} 
};


static PyObject* FiniteRotation_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    FiniteRot *self;
    Covariance *tmp;
    PyObject *lon_obj, *lat_obj, *angle_obj, *time_obj, *cov = NULL;
    
    double lon =0.0, lat = 0.0, angle = 0.0, time = 0.0;
    cov = Covariance_new(&CovarianceType, PyTuple_New(0), NULL);


    if (kwds != NULL && PyDict_Size(kwds) > 0) {
        PyErr_SetString(PyExc_TypeError, "FiniteRotation() does not accept keyword arguments");
        return NULL;
    }
    
    if (PyTuple_Size(args) == 0 || args == NULL){
    } else if (PyTuple_Size(args) == 5 || PyTuple_Size(args) == 4) {

        if (!PyArg_ParseTuple(args, "OOOO|O", &lon_obj, &lat_obj, &angle_obj, &time_obj, &cov)) {
            return NULL;
        }

        // Check if the first four arguments are doubles
        if (!PyFloat_Check(lon_obj) || !PyFloat_Check(lat_obj) ||
            !PyFloat_Check(angle_obj) || !PyFloat_Check(time_obj)) {
            Py_XDECREF(lon_obj);
            Py_XDECREF(lat_obj);
            Py_XDECREF(angle_obj);
            Py_XDECREF(time_obj);
            Py_XDECREF(cov);
            PyErr_SetString(PyExc_TypeError, "Lon, Lat, Angle and Time arguments must be doubles");
            return NULL;

        } else {
            // Convert the doubles
            lon = PyFloat_AsDouble(lon_obj);
            lat = PyFloat_AsDouble(lat_obj);
            angle = PyFloat_AsDouble(angle_obj);
            time = PyFloat_AsDouble(time_obj);
        }

        if (!PyObject_IsInstance(cov, (PyObject *)&CovarianceType)) {
            Py_XDECREF(lon_obj);
            Py_XDECREF(lat_obj);
            Py_XDECREF(angle_obj);
            Py_XDECREF(time_obj);
            Py_XDECREF(cov);
            PyErr_SetString(PyExc_TypeError, "Covariance argument must be of type Covariance()");
            return NULL;

        }
    }
    else {
        Py_XDECREF(cov);
        PyErr_SetString(PyExc_TypeError, "FiniteRotation() constructor accepts at four or five arguments: Lon, Lat, Angle, Time, *Covariance");
        return NULL;
    }

    self = (FiniteRot *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->Lon = lon;
        self->Lat = lat;
        self->Angle = angle;
        self->Time = time;

        tmp = &(self->Covariance);
        Py_INCREF(cov);
        self->Covariance = *((Covariance *)cov);
        Py_XDECREF(tmp);
    }
    return (PyObject *)self;
}


// Representation of FiniteRotation object
static PyObject* FiniteRotation_repr(FiniteRot *self) {
    const char *format = "FiniteRot(Lon=%.2f, Lat=%.2f, Angle=%.3f, Time=%.3f)";
    char buffer[256];  // Adjust the buffer size as needed

    snprintf(buffer, sizeof(buffer), format,
             self->Lon, self->Lat, self->Angle, self->Time);

    return PyUnicode_FromString(buffer);
}


static void FiniteRotation_dealloc(FiniteRot *self) {
    // First, deallocate the nested Covariance object
    Covariance_dealloc(&(self->Covariance));

    // Then, deallocate the FiniteRot object itself
    Py_TYPE(self)->tp_free(self);
}


// Getter functions for FiniteRotation attributes
static PyObject* FiniteRotation_get_Lon(FiniteRot *self, void *closure) {
    return PyFloat_FromDouble(self->Lon);
}

static PyObject* FiniteRotation_get_Lat(FiniteRot *self, void *closure) {
    return PyFloat_FromDouble(self->Lat);
}

static PyObject* FiniteRotation_get_Angle(FiniteRot *self, void *closure) {
    return PyFloat_FromDouble(self->Angle);
}

static PyObject* FiniteRotation_get_Time(FiniteRot *self, void *closure) {
    return PyFloat_FromDouble(self->Time);
}

static PyObject* FiniteRotation_get_Covariance(FiniteRot *self, void *closure) {
/*     PyObject *covariance_args = PyTuple_New(0); 
    PyObject *covariance_instance = Covariance_new(&CovarianceType, covariance_args, NULL);
    Py_DECREF(covariance_args);  

    if (covariance_instance == NULL) {
        return NULL;
    }

    Covariance *covariance = (Covariance *)covariance_instance;
    covariance->C11 = self->Covariance.C11;
    covariance->C12 = self->Covariance.C12;
    covariance->C13 = self->Covariance.C13;
    covariance->C22 = self->Covariance.C22;
    covariance->C23 = self->Covariance.C23;
    covariance->C33 = self->Covariance.C33; 

    return (PyObject *)covariance; */
    Covariance *cov = &(self->Covariance);
    Py_INCREF(cov);
    //return self->first;
    //Covariance *cov_instance = &(self->Covariance);
    return (PyObject *)cov;
} 


// Setter functions for FiniteRotation attributes
static int FiniteRotation_set_Lon(FiniteRot *self, PyObject *value, void *closure) {
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

static int FiniteRotation_set_Lat(FiniteRot *self, PyObject *value, void *closure) {
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

static int FiniteRotation_set_Angle(FiniteRot *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
        return -1;
    }
    if (!PyFloat_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Angle attribute must be a float");
        return -1;
    }
    self->Angle = PyFloat_AsDouble(value);
    return 0;
}

static int FiniteRotation_set_Time(FiniteRot *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
        return -1;
    }
    if (!PyFloat_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Time attribute must be a float");
        return -1;
    }
    self->Time = PyFloat_AsDouble(value);
    return 0;
}

static int FiniteRotation_set_Covariance(FiniteRot *self, PyObject *cov_instance, void *closure) {
    if (!PyObject_TypeCheck(cov_instance, &CovarianceType)) {
        PyErr_SetString(PyExc_TypeError, "Covariance attribute must be a Covariance instance");
        return -1;
    }

    Covariance *cov = &(self->Covariance);
    Py_DECREF((PyObject *)cov);
    Py_INCREF(cov_instance);
    self->Covariance = *((Covariance *)cov_instance);
/*     Covariance *covariance = (Covariance *)cov_instance;

    self->Covariance.C11 = covariance->C11;
    self->Covariance.C12 = covariance->C12;
    self->Covariance.C13 = covariance->C13;
    self->Covariance.C22 = covariance->C22;
    self->Covariance.C23 = covariance->C23;
    self->Covariance.C33 = covariance->C33; */

    return 0;
} 

static PyGetSetDef FiniteRotation_getsetters[] = {
    {"Lon", (getter)FiniteRotation_get_Lon, (setter)FiniteRotation_set_Lon, "Longitude", NULL},
    {"Lat", (getter)FiniteRotation_get_Lat, (setter)FiniteRotation_set_Lat, "Latitude", NULL},
    {"Angle", (getter)FiniteRotation_get_Angle, (setter)FiniteRotation_set_Angle, "Angle", NULL},
    {"Time", (getter)FiniteRotation_get_Time, (setter)FiniteRotation_set_Time, "Time", NULL},
    {"Covariance", (getter)FiniteRotation_get_Covariance, (setter)FiniteRotation_set_Covariance, "Covariance attribute", NULL},
    {NULL}
};


static PyMethodDef FiniteRotation_methods[] = {
    {"build_ensemble", py_build_fr_ensemble, METH_VARARGS, "Draws n FiniteRotation() samples from the covariance of a given finite rotation."},
    {NULL, NULL, 0, NULL}
};


static PyTypeObject FiniteRotationType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pk_structs.FiniteRotation",
    .tp_doc = "FiniteRotation object",
    .tp_members = FiniteRotation_members,
    .tp_repr = (reprfunc)FiniteRotation_repr,
    .tp_basicsize = sizeof(FiniteRot),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = FiniteRotation_new,
    .tp_dealloc = (destructor)FiniteRotation_dealloc,
    .tp_getset = FiniteRotation_getsetters,
    .tp_methods = FiniteRotation_methods,
};