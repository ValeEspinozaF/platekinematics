#define PY_ARRAY_UNIQUE_SYMBOL PLATEKIN_ARRAY_API
#include "pk_structs.h"

PyObject* py_fr_average(PyObject *self, PyObject *args);
PyObject *py_ev_average(PyObject *self, PyObject *args);


/* Methods contained in the module */
static PyMethodDef methodsMethods[] = {
    {"average_fr", py_fr_average, METH_VARARGS, "average_fr(ensemble, time=0.0) -> FiniteRotation\n\nReturn the average finite rotation from either a list of\nFiniteRotation objects or a NumPy array of rotation matrices\nwith shape (n_size, 3, 3). The optional time argument sets the\nTime value on the returned object."},
    {"average_ev", py_ev_average, METH_VARARGS, "average_ev(ensemble, time_range=(0.0, 0.0)) -> EulerVector\n\nReturn the average Euler vector from either a NumPy array of\nCartesian coordinates with shape (3, n_size) or a list of\nEulerVector samples. The optional time_range is stored on the\nreturned EulerVector."},
  {NULL, NULL, 0, NULL}
};

static PyModuleDef pk_structs = {
    PyModuleDef_HEAD_INIT,
    .m_name = "pk_structs",
    .m_doc = "Plate kinematics structures and ensemble utilities.\n\nExports Covariance, FiniteRotation, EulerVector, average_fr, and average_ev.",
    .m_size = -1,
    .m_methods = methodsMethods,
};


PyMODINIT_FUNC PyInit_pk_structs(void) {
    PyObject *m;
    import_array(); // Initialize NumPy API

    if (PyType_Ready(&CovarianceType) < 0)
        return NULL;

    if (PyType_Ready(&FiniteRotationType) < 0)
        return NULL;

    if (PyType_Ready(&EulerVectorType) < 0)
        return NULL;

    m = PyModule_Create(&pk_structs);
    if (m == NULL)
        return NULL;

    Py_INCREF(&CovarianceType);
    PyModule_AddObject(m, "Covariance", (PyObject *)&CovarianceType);

    Py_INCREF(&FiniteRotationType);
    PyModule_AddObject(m, "FiniteRotation", (PyObject *)&FiniteRotationType);

    Py_INCREF(&EulerVectorType);
    PyModule_AddObject(m, "EulerVector", (PyObject *)&EulerVectorType);
    return m;
}