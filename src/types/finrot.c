#include <Python.h>
#include "finrot.h"


static PyObject* FiniteRotation_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    FiniteRot *self;

    if (kwds != NULL && PyDict_Size(kwds) > 0) {
        PyErr_SetString(PyExc_TypeError, "FiniteRot_new() does not accept keyword arguments");
        return NULL;
    }
    
    self = (FiniteRot *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->Lon = 0.0;
        self->Lat = 0.0;
        self->Angle = 0.0; 
        self->Time = 0.0;


        PyObject *cov_args = PyTuple_New(0);
        PyObject *cov_instance = Covariance_new(&CovarianceType, cov_args, NULL);
        Py_DECREF(cov_args); 
        if (cov_instance == NULL) {
            Py_DECREF(self);
            return NULL;
        }

        self->Covariance = *((Covariance *)cov_instance);  
        Py_DECREF(cov_instance); 
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
    Py_TYPE(self)->tp_free((PyObject *)self);
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
    //PyObject *covariance_args = PyTuple_New(0); 
    //PyFloat_FromDouble(self->C11)
    PyObject *covariance_instance = Covariance_new(&CovarianceType, NULL, NULL);
    //Py_DECREF(covariance_args);  

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

    return covariance_instance;
    //Covariance *cov_instance = &(self->Covariance);
    //return (PyObject *)cov_instance;
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

    Covariance *covariance = (Covariance *)cov_instance;
    self->Covariance.C11 = covariance->C11;
    self->Covariance.C12 = covariance->C12;
    self->Covariance.C13 = covariance->C13;
    self->Covariance.C22 = covariance->C22;
    self->Covariance.C23 = covariance->C23;
    self->Covariance.C33 = covariance->C33;

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

static PyTypeObject FiniteRotationType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pk_structs.FiniteRotation",
    .tp_doc = "FiniteRotation object",
    .tp_repr = (reprfunc)FiniteRotation_repr,
    .tp_basicsize = sizeof(FiniteRot),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = FiniteRotation_new,
    .tp_dealloc = (destructor)FiniteRotation_dealloc,
    .tp_getset = FiniteRotation_getsetters,
    //.tp_methods = FiniteRotation_methods,
};