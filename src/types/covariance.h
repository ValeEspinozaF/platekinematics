#ifndef COVARIANCE_H
#define COVARIANCE_H

#include <Python.h>

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
void Covariance_dealloc(Covariance *self);
extern PyTypeObject CovarianceType;


#endif // COVARIANCE_H