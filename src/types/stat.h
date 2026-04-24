#ifndef STAT_H
#define STAT_H

#include "../platekinematics.h"

typedef struct {
    PyObject_HEAD
    double Mean;
    double StDev;
} Stat;

extern PyTypeObject StatType;

#endif // STAT_H