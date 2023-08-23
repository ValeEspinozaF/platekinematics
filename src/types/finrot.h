#include <Python.h> 
#include "covariance.h"

#ifndef FINITE_ROT_H
#define FINITE_ROT_H

typedef struct {
    PyObject_HEAD
    double Lon;   // Longitude of the rotation axis in degrees-East.
    double Lat;   // Latitude of the rotation axis in degrees-North.
    double Angle; // Angle of rotation in degrees.
    double Time;  // Age of rotation in million years.
    Covariance Covariance;  // Covariance in radians².
} FiniteRot;

#endif // FINITE_ROT_H