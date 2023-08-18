#include <Python.h>
#include "myobject.h"


static PyObject* MyObject_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    MyObject *self;
    self = (MyObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->x = 0;
        self->y = 0.0;
    }
    return (PyObject *)self;
}

static void MyObject_dealloc(MyObject *self) {
    Py_TYPE(self)->tp_free((PyObject *)self);
}


static PyObject* MyObject_get_x(MyObject *self, void *closure) {
    return PyLong_FromLong(self->x);
}

static int MyObject_set_x(MyObject *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
        return -1;
    }
    if (!PyLong_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Attribute value must be an integer");
        return -1;
    }
    self->x = PyLong_AsLong(value);
    return 0;
}

static PyObject* MyObject_get_y(MyObject *self, void *closure) {
    return PyFloat_FromDouble(self->y);
}

static int MyObject_set_y(MyObject *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
        return -1;
    }
    if (!PyFloat_Check(value)) {
        PyErr_SetString(PyExc_TypeError, "Attribute value must be a float");
        return -1;
    }
    self->y = PyFloat_AsDouble(value);
    return 0;
}

static PyGetSetDef MyObject_getsetters[] = {
    {"x", (getter)MyObject_get_x, (setter)MyObject_set_x, "Integer attribute", NULL},
    {"y", (getter)MyObject_get_y, (setter)MyObject_set_y, "Double attribute", NULL},
    {NULL}
};


static PyTypeObject MyObjectType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "myextension.MyObject",
    .tp_doc = "MyObject objects",
    .tp_basicsize = sizeof(MyObject),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = MyObject_new,
    .tp_dealloc = (destructor)MyObject_dealloc,
    .tp_getset = MyObject_getsetters,
};

/*
static PyModuleDef myextension_module = {
    PyModuleDef_HEAD_INIT,
    .m_name = "myextension",
    .m_doc = "Example module",
    .m_size = -1,
};

PyMODINIT_FUNC PyInit_myextension_module(void) {
    PyObject *m;

    if (PyType_Ready(&MyObjectType) < 0)
        return NULL;

    m = PyModule_Create(&myextension_module);
    if (m == NULL)
        return NULL;

    Py_INCREF(&MyObjectType);
    PyModule_AddObject(m, "MyObject", (PyObject *)&MyObjectType);
    return m;
}
*/
