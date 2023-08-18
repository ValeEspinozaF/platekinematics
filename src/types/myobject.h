#include <Python.h>

#ifndef MYOBJECT_H
#define MYOBJECT_H

typedef struct {
    PyObject_HEAD
    int x;
    double y;
} MyObject;

#endif  // MYOBJECT_H
