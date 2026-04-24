#ifndef SURFACE_VELOCITY_H
#define SURFACE_VELOCITY_H

#include "../platekinematics.h"
#include "stat.h"

typedef struct {
    PyObject_HEAD
    double Lon;
    double Lat;
    PyObject *EastVel;
    PyObject *NorthVel;
    PyObject *TotalVel;
    PyObject *Azimuth;
} SurfaceVelocity;

extern PyTypeObject SurfaceVelocityType;

#endif // SURFACE_VELOCITY_H