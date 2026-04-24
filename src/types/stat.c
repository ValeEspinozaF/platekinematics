#include "../pk_structs.h"

static bool stat_is_number(PyObject *value) {
    return PyFloat_Check(value) || PyLong_Check(value);
}

static int stat_set_numeric(double *target, PyObject *value, const char *field_name) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
        return -1;
    }
    if (!stat_is_number(value)) {
        PyErr_Format(PyExc_TypeError, "%s attribute must be a float", field_name);
        return -1;
    }
    *target = PyFloat_AsDouble(value);
    return 0;
}

static PyObject* Stat_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    PyObject *first = NULL;
    PyObject *second = NULL;
    double mean = 0.0;
    double stdev = 0.0;

    if (kwds != NULL && PyDict_Size(kwds) > 0) {
        PyErr_SetString(PyExc_TypeError, "Stat() does not accept keyword arguments");
        return NULL;
    }

    if (args == NULL || PyTuple_Size(args) == 0) {
        mean = 0.0;
        stdev = 0.0;

    } else if (PyTuple_Size(args) == 1) {
        double *values;
        if (!PyArg_ParseTuple(args, "O", &first)) {
            return NULL;
        }

        values = parse_double_array(first);
        if (values == NULL) {
            return NULL;
        }
        if ((int)values[0] != 2) {
            free(values);
            PyErr_SetString(PyExc_ValueError, "Stat() single-argument constructor expects exactly 2 values");
            return NULL;
        }

        mean = values[1];
        stdev = values[2];
        free(values);

    } else if (PyTuple_Size(args) == 2) {
        if (!PyArg_ParseTuple(args, "OO", &first, &second)) {
            PyErr_SetString(PyExc_TypeError, "Stat() failed to parse one or more input arguments");
            return NULL;
        }
        if (!stat_is_number(first) || !stat_is_number(second)) {
            PyErr_SetString(PyExc_TypeError, "Stat() constructor expects two floats");
            return NULL;
        }

        mean = PyFloat_AsDouble(first);
        stdev = PyFloat_AsDouble(second);

    } else {
        PyErr_SetString(PyExc_TypeError, "Stat() constructor accepts zero arguments, one 2-value array-like argument, or two floats");
        return NULL;
    }

    Stat *self = (Stat *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->Mean = mean;
        self->StDev = stdev;
    }
    return (PyObject *)self;
}

static PyObject* Stat_repr(Stat *self) {
    char buffer[128];
    snprintf(buffer, sizeof(buffer), "Stat(Mean=%.6g, StDev=%.6g)", self->Mean, self->StDev);
    return PyUnicode_FromString(buffer);
}

static int Stat_set_Mean(Stat *self, PyObject *value, void *closure) {
    return stat_set_numeric(&self->Mean, value, "Mean");
}

static int Stat_set_StDev(Stat *self, PyObject *value, void *closure) {
    return stat_set_numeric(&self->StDev, value, "StDev");
}

static PyGetSetDef Stat_getsetters[] = {
    {"Mean", NULL, (setter)Stat_set_Mean, "Mean (average).", NULL},
    {"StDev", NULL, (setter)Stat_set_StDev, "Standard deviation.", NULL},
    {NULL}
};

static PyMemberDef Stat_members[] = {
    {"Mean", T_DOUBLE, offsetof(Stat, Mean), 0, "Mean (average)."},
    {"StDev", T_DOUBLE, offsetof(Stat, StDev), 0, "Standard deviation."},
    {NULL}
};

PyTypeObject StatType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pk_structs.Stat",
    .tp_doc = "Stat(mean=0.0, stdev=0.0)\n\nMean and standard deviation container.\n\nAccepted constructors:\n- Stat(mean, stdev)\n- Stat([mean, stdev])",
    .tp_basicsize = sizeof(Stat),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = Stat_new,
    .tp_repr = (reprfunc)Stat_repr,
    .tp_members = Stat_members,
    .tp_getset = Stat_getsetters,
};