#include <Python.h>




/* Methods contained in the module */
static PyMethodDef platekinematicsMethods[] = {
  {NULL, NULL, 0, NULL}
};


/* Module definition */
static struct PyModuleDef platekinematics = {
  PyModuleDef_HEAD_INIT,
  "platekinematics",
  "Custom maths module",
  -1,
  platekinematicsMethods
};

PyMODINIT_FUNC PyInit_platekinematics(void)
{
    return PyModule_Create(&platekinematics);
}
