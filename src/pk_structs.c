#include <Python.h>
#include "types/covariance.c"
#include "types/finrot.c"


static PyModuleDef pk_structs = {
    PyModuleDef_HEAD_INIT,
    .m_name = "pk_structs",
    .m_doc = "Plate kinematics structures.",
    .m_size = -1,
};


PyMODINIT_FUNC PyInit_pk_structs(void) {
    PyObject *m;

    if (PyType_Ready(&CovarianceType) < 0)
        return NULL;

    if (PyType_Ready(&FiniteRotationType) < 0)
        return NULL;

    m = PyModule_Create(&pk_structs);
    if (m == NULL)
        return NULL;

    Py_INCREF(&CovarianceType);
    PyModule_AddObject(m, "Covariance", (PyObject *)&CovarianceType);

    Py_INCREF(&FiniteRotationType);
    PyModule_AddObject(m, "FiniteRotation", (PyObject *)&FiniteRotationType);
    return m;
}