#include <Python.h> 
#include "covariance.h"

#ifndef EULER_VECTOR_H
#define EULER_VECTOR_H

typedef struct {
    PyObject_HEAD
    double Lon;           // Longitude of the Euler pole in degrees-East.
    double Lat;           // Latitude of the Euler pole in degrees-North.
    double AngVelocity;   // Angular velocity in degrees/Myr.
    double TimeRange[2];     // Initial to final age of rotation.
    Covariance Covariance; // Covariance in radians²/Myr².
    int has_covariance;
} EulerVector;

/* struct EulerVectorCart {
    double X;             // X-coordinate in degrees/Myr.
    double Y;             // Y-coordinate in degrees/Myr.
    double Z;             // Z-coordinate in degrees/Myr.
    double (*TimeRange)[2];     // Initial to final age of rotation.
    struct Covariance Covariance; // Covariance in radians²/Myr².
}; */

#endif // EULER_VECTOR_H