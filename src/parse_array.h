#ifndef PARSE_ARRAY_H
#define PARSE_ARRAY_H

#include "platekinematics.h"

gsl_matrix* pyarray2D_to_gslmatrix(PyObject *pyarray);
gsl_matrix** pyarray3D_to_gslmatrix(PyObject *pyarray);
double* parse_tuple(PyObject *py_obj);
double* parse_list(PyObject *py_obj);
double* parse_numpy_1Darray(PyObject *py_obj);
double* parse_double_array(PyObject *py_obj);
PyObject* build_numpy_3Darray(gsl_matrix **cA, int dim0_n_size);
PyObject * build_numpy_2Darray(gsl_matrix *cA);

#endif // PARSE_ARRAY_H