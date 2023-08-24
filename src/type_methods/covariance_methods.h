#ifndef COV_METHODS_H
#define COV_METHODS_H

#include "../types/covariance.h"
#include "../gsl.h"

gsl_matrix * to_matrix(Covariance *cov);

#endif // COV_METHODS_H