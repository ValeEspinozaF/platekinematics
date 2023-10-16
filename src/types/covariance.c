#include "../pk_structs.h"

PyObject *py_cov_to_numpy(PyObject *self, int Py_UNUSED(_));

static PyMemberDef Covariance_members[] = {
    {"C11", T_DOUBLE, offsetof(Covariance, C11), 0, "First element of the covariance matrix"},
    {"C12", T_DOUBLE, offsetof(Covariance, C12), 0, "Second element of the covariance matrix"},
    {"C13", T_DOUBLE, offsetof(Covariance, C13), 0, "Third element of the covariance matrix"},
    {"C22", T_DOUBLE, offsetof(Covariance, C22), 0, "Fourth element of the covariance matrix"},
    {"C23", T_DOUBLE, offsetof(Covariance, C23), 0, "Fifth element of the covariance matrix"},
    {"C33", T_DOUBLE, offsetof(Covariance, C33), 0, "Sixth element of the covariance matrix"},
    {NULL} 
};

static PyObject* Covariance_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Covariance *self;
    PyObject *py_obj;
    double *arg_list;

    if (kwds != NULL && PyDict_Size(kwds) > 0) {
        PyErr_SetString(PyExc_TypeError, "Covariance() does not accept keyword arguments");
        return NULL;
    }

    if (PyTuple_Size(args) == 0 || args == NULL){
        arg_list = (double *)malloc((6 + 1) * sizeof(double));
        arg_list[0] = 6.0;
        double list[] = {1.0, 0.0, 0.0, 1.0, 0.0, 1.0}; // default constructor
        for (int i = 0; i < 6; i++) {
                arg_list[i + 1] = list[i];
            }

    } else if (PyTuple_Size(args) == 1) {
        if (!PyArg_ParseTuple(args, "O", &py_obj)) {
            return NULL;
        }

        arg_list = parse_double_array(py_obj); // args constructor
        if (arg_list == NULL) {
            return NULL;
        } 

        if ((int)arg_list[0] != 6) {
            PyErr_SetString(PyExc_ValueError, "Constructor argument must contain exactly 6 elements");
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "Covariance() constructor accepts at most one argument, a list of 6 doubles");
        return NULL;
    }

    self = (Covariance *)type->tp_alloc(type, 0);
    if (self != NULL) {

        for (int i = 0; i < 6; ++i) {
            double value = arg_list[i + 1];

            switch (i) {
                case 0: self->C11 = value; break;
                case 1: self->C12 = value; break;
                case 2: self->C13 = value; break;
                case 3: self->C22 = value; break;
                case 4: self->C23 = value; break;
                case 5: self->C33 = value; break;
            }
        }
    }

    free(arg_list);
    return (PyObject *)self;
}

// Representation of Covariance object
static PyObject* Covariance_repr(Covariance *self) {
    char buffer[100];
    const char *format = "Covariance(C11=%.2e, C12=%.2e, C13=%.2e, C22=%.2e, C23=%.2e, C33=%.2e)";
    snprintf(buffer, sizeof(buffer), format,
             self->C11, self->C12, self->C13, self->C22, self->C23, self->C33);

    return PyUnicode_FromString(buffer);
}


void Covariance_dealloc(Covariance *self) {
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


static PyMethodDef Covariance_methods[] = {
    {"to_numpy", py_cov_to_numpy, METH_NOARGS, "Converts a Covariance() instance to a numpy array."},
    {NULL, NULL, 0, NULL}
};

PyTypeObject CovarianceType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pk_structs.Covariance",
    .tp_doc = "Covariance object with attributes C11-C33",
    .tp_members = Covariance_members,
    .tp_repr = (reprfunc)Covariance_repr,
    .tp_basicsize = sizeof(Covariance),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Covariance_new,
    .tp_dealloc = (destructor)Covariance_dealloc,
    .tp_getset = Covariance_getsetters,
    .tp_methods = Covariance_methods,
};