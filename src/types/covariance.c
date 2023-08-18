#include <Python.h>
#include "covariance.h"


static PyObject* Covariance_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Covariance *self;
    self = (Covariance *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->C11 = 0.0;
        self->C12 = 0.0;
        self->C13 = 0.0;
        self->C22 = 0.0;
        self->C23 = 0.0;
        self->C33 = 0.0;
    }
    return (PyObject *)self;
}

static void Covariance_dealloc(Covariance *self) {
    Py_TYPE(self)->tp_free((PyObject *)self);
}


// Getter functions for Covariance attributes
static PyObject* Covariance_get_C11(Covariance *self, void *closure) {
    return PyFloat_FromDouble(self->C11);
}
static PyObject* Covariance_get_C12(Covariance *self, void *closure) {
    return PyFloat_FromDouble(self->C12);
}
static PyObject* Covariance_get_C13(Covariance *self, void *closure) {
    return PyFloat_FromDouble(self->C13);
}
static PyObject* Covariance_get_C22(Covariance *self, void *closure) {
    return PyFloat_FromDouble(self->C22);
}
static PyObject* Covariance_get_C23(Covariance *self, void *closure) {
    return PyFloat_FromDouble(self->C23);
}
static PyObject* Covariance_get_C33(Covariance *self, void *closure) {
    return PyFloat_FromDouble(self->C33);
}


// Setter functions for Covariance attributes
static int Covariance_set_C11(Covariance *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
        return -1;
    }
    if (!PyFloat_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "C11 attribute must be a float");
        return -1;
    }
    self->C11 = PyFloat_AsDouble(value);
    return 0;
}
static int Covariance_set_C12(Covariance *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
        return -1;
    }
    if (!PyFloat_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "C12 attribute must be a float");
        return -1;
    }
    self->C12 = PyFloat_AsDouble(value);
    return 0;
}
static int Covariance_set_C13(Covariance *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
        return -1;
    }
    if (!PyFloat_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "C13 attribute must be a float");
        return -1;
    }
    self->C13 = PyFloat_AsDouble(value);
    return 0;
}
static int Covariance_set_C22(Covariance *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
        return -1;
    }
    if (!PyFloat_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "C22 attribute must be a float");
        return -1;
    }
    self->C22 = PyFloat_AsDouble(value);
    return 0;
}
static int Covariance_set_C23(Covariance *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
        return -1;
    }
    if (!PyFloat_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "C23 attribute must be a float");
        return -1;
    }
    self->C23 = PyFloat_AsDouble(value);
    return 0;
}
static int Covariance_set_C33(Covariance *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
        return -1;
    }
    if (!PyFloat_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "C33 attribute must be a float");
        return -1;
    }
    self->C33 = PyFloat_AsDouble(value);
    return 0;
}


static PyGetSetDef Covariance_getsetters[] = {
    {"C11", (getter)Covariance_get_C11, (setter)Covariance_set_C11, "C11 attribute", NULL},
    {"C12", (getter)Covariance_get_C12, (setter)Covariance_set_C12, "C12 attribute", NULL},
    {"C13", (getter)Covariance_get_C13, (setter)Covariance_set_C13, "C13 attribute", NULL},
    {"C22", (getter)Covariance_get_C22, (setter)Covariance_set_C22, "C22 attribute", NULL},
    {"C23", (getter)Covariance_get_C23, (setter)Covariance_set_C23, "C23 attribute", NULL},
    {"C33", (getter)Covariance_get_C33, (setter)Covariance_set_C33, "C33 attribute", NULL},
    {NULL}
};

void set_all_to_value(Covariance *cov, double value) {
    cov->C11 = value;
    cov->C12 = value;
    cov->C13 = value;
    cov->C22 = value;
    cov->C23 = value;
    cov->C33 = value;
}


static PyObject* py_set_all_to_value(PyObject *self, PyObject *args) {
    Covariance *cov = (Covariance*)self;
    double value;

    if (!PyArg_ParseTuple(args, "d", &value)) {
        return NULL;
    }

    // Create a new instance of the Covariance type
    PyObject *covariance_type = PyObject_GetAttrString((PyObject*)self, "__class__");
    if (covariance_type == NULL) {
        return NULL;
    }

    Covariance *new_cov = (Covariance*)PyObject_CallObject(covariance_type, NULL);
    Py_DECREF(covariance_type);
    if (new_cov == NULL) {
        return NULL;
    }

    set_all_to_value(new_cov, value);

    return (PyObject*)new_cov;
}

static PyMethodDef Covariance_methods[] = {
    {"set_all_to_value", py_set_all_to_value, METH_VARARGS, "Set all attributes to a given value"},
    {NULL, NULL, 0, NULL} 
};


static PyTypeObject CovarianceType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "covariance_test.Covariance",
    .tp_doc = "Covariance objects",
    .tp_basicsize = sizeof(Covariance),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Covariance_new,
    .tp_dealloc = (destructor)Covariance_dealloc,
    .tp_getset = Covariance_getsetters,
    .tp_methods = Covariance_methods,
};