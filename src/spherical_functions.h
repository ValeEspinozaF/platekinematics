#ifndef SPHERICAL_FUNCTIONS_H
#define SPHERICAL_FUNCTIONS_H

#include "platekinematics.h"

/* Declarations for the functions in the module */
inline double to_degrees(double radians);
inline double to_radians(double degrees);
double* sph2cart(double lon, double lat, double mag);
double* cart2sph(double x, double y, double z);

#endif // SPHERICAL_FUNCTIONS_H