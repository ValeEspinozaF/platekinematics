#include <Python.h>

#ifndef COVARIANCE_H
#define COVARIANCE_H

typedef struct {
    PyObject_HEAD
    double C11;
    double C12;
    double C13;
    double C22;
    double C23;
    double C33;
} Covariance;

void set_all_to_value(Covariance *cov, double value);

#endif // COVARIANCE_H