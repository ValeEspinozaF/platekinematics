#ifndef COV_METHODS_H
#define COV_METHODS_H

#include "../types/covariance.h"
#include "../types/eulervec.h"
#include "../types/finrot.h"
#include "../gsl.h"

gsl_matrix * to_matrix(Covariance *cov);
gsl_matrix * ev_cov_to_matrix(EulerVector *ev_sph);
gsl_matrix * fr_cov_to_matrix(FiniteRot *fr_sph);

#endif // COV_METHODS_H