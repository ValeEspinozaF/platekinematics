#ifndef EULERVEC_METHODS_H
#define EULERVEC_METHODS_H

#include <Python.h>
#include "../gsl.h"
#include "../types/eulervec.h"

gsl_matrix * build_ev_array(EulerVector *ev_sph, int n_size, const char* coordinate_system);


#endif // EULERVEC_METHODS_H