#define PY_ARRAY_UNIQUE_SYMBOL PLATEKIN_ARRAY_API
#include "pk_structs.h"

PyObject* py_fr_average(PyObject *self, PyObject *args);
PyObject *py_ev_average(PyObject *self, PyObject *args);


/* Methods contained in the module */
static PyMethodDef methodsMethods[] = {
  {"average_fr", py_fr_average, METH_VARARGS, "Return the average finite rotation from a given ensemble of rotation matrices [units]."},
  {"average_ev", py_ev_average, METH_VARARGS, "Return the average vector from a given ensemble of x, y and z vector coordinates in [units]."},
  {NULL, NULL, 0, NULL}
};


/* Module definition */
static struct PyModuleDef methods = {
  PyModuleDef_HEAD_INIT,
  "methods",
  "Custom methods module",
  -1,
  methodsMethods
};

PyMODINIT_FUNC PyInit_methods(void){
    PyObject *m;
    import_array(); // Initialize NumPy API

/*     if (PyType_Ready(&CovarianceType) < 0)
        return NULL;

    if (PyType_Ready(&EulerVectorType) < 0)
        return NULL; */

    m = PyModule_Create(&methods);
    if (m == NULL)
        return NULL;

/*     Py_INCREF(&CovarianceType);
    PyModule_AddObject(m, "Covariance", (PyObject *)&CovarianceType);

    Py_INCREF(&FiniteRotationType);
    PyModule_AddObject(m, "FiniteRotation", (PyObject *)&FiniteRotationType);

    Py_INCREF(&EulerVectorType);
    PyModule_AddObject(m, "EulerVector", (PyObject *)&EulerVectorType); */
    
    return m;
}