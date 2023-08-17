#ifndef FINITE_ROT_H
#define FINITE_ROT_H

#include "covariance.h"

struct FiniteRotSph {
    double Lon;   // Longitude of the rotation axis in degrees-East.
    double Lat;   // Latitude of the rotation axis in degrees-North.
    double Angle; // Angle of rotation in degrees.
    double Time;  // Age of rotation in million years.
    struct Covariance Covariance;  // Covariance in radians².
};

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