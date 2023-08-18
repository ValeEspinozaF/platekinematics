#include <Python.h>
#include "covariance.h"

#ifndef FINITE_ROT_H
#define FINITE_ROT_H

typedef struct {
    double Lon;   // Longitude of the rotation axis in degrees-East.
    double Lat;   // Latitude of the rotation axis in degrees-North.
    double Angle; // Angle of rotation in degrees.
    double Time;  // Age of rotation in million years.
    Covariance Covariance;  // Covariance in radians².
} FiniteRotSph;

/*
struct FiniteRotCart {
    double X;      // X-coordinate in degrees.
    double Y;      // Y-coordinate in degrees.
    double Z;      // Z-coordinate in degrees.
    double Time;   // Age of rotation in million years.
    struct Covariance Covariance;  // Covariance in radians².
};
*/

#endif // FINITE_ROT_H