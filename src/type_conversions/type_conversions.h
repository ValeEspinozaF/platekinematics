#ifndef CONVERSIONS_H
#define CONVERSIONS_H

#include "../pk_structs.h"
#include "../spherical_functions.h"

gsl_matrix * cov_to_matrix(Covariance *cov);
gsl_matrix * ev_cov_to_matrix(EulerVector *ev_sph);
gsl_matrix * fr_cov_to_matrix(FiniteRot *fr_sph);

gsl_matrix* fr_to_rotation_matrix(const FiniteRot *fr_sph);
FiniteRot* rotation_matrix_to_fr(gsl_matrix* m);
gsl_matrix* ea_to_rotation_matrix(double* ea);
gsl_matrix** eas_to_rotation_matrices(gsl_vector* ex, gsl_vector* ey, gsl_vector* ez);
double* rotation_matrix_to_ea(gsl_matrix* m);
gsl_matrix* rotation_matrices_to_eas(gsl_matrix** m_array, int dim0_n_size);
double* fr_to_euler_angles(const FiniteRot *fr_sph);
FiniteRot* ea_to_finrot(double* ea);


#endif // CONVERSIONS_H