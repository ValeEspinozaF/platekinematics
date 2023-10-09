#include "type_conversions.h"

// Return a 3x3 rotation matrix (expressed in radians) from a finite rotation (expressed in degrees).
gsl_matrix* fr_to_rotation_matrix(const FiniteRot *fr_sph) {
    double x, y, z; 

    double *r = sph2cart(fr_sph->Lon, fr_sph->Lat, fr_sph->Angle);
    x = r[0];
    y = r[1];
    z = r[2];

    double a = to_radians(fr_sph->Angle);
    double b = 1.0 - cos(a);
    double c = sin(a);

    gsl_matrix *m = gsl_matrix_alloc(3, 3);

    // Rotation matrix in [radians]
    gsl_matrix_set(m, 0, 0, cos(a) + x * x * b);
    gsl_matrix_set(m, 0, 1, x * y * b - z * c);
    gsl_matrix_set(m, 0, 2, x * z * b + y * c);
    gsl_matrix_set(m, 1, 0, y * x * b + z * c);
    gsl_matrix_set(m, 1, 1, cos(a) + y * y * b);
    gsl_matrix_set(m, 1, 2, y * z * b - x * c);
    gsl_matrix_set(m, 2, 0, z * x * b - y * c);
    gsl_matrix_set(m, 2, 1, z * y * b + x * c);
    gsl_matrix_set(m, 2, 2, cos(a) + z * z * b);

    return m;
}

//Return a finite rotation (expressed in degrees) from a 3x3 rotation matrix (expressed in radians).
FiniteRot* rotation_matrix_to_fr(gsl_matrix* m) {
    double x = gsl_matrix_get(m, 2, 1) - gsl_matrix_get(m, 1, 2);
    double y = gsl_matrix_get(m, 0, 2) - gsl_matrix_get(m, 2, 0);
    double z = gsl_matrix_get(m, 1, 0) - gsl_matrix_get(m, 0, 1);

    double* fr_sph = cart2sph(x, y, z);
    double r = sqrt(pow(x,2) + pow(y,2) + pow(z,2));
    double t = gsl_matrix_get(m, 0, 0) + gsl_matrix_get(m, 1, 1) + gsl_matrix_get(m, 2, 2);
    double mag = to_degrees(atan2(r, t - 1) );
    
    FiniteRot *fr = PyObject_New(FiniteRot, &FiniteRotationType);
    if (fr != NULL) {
        fr->Lon = fr_sph[0];
        fr->Lat = fr_sph[1];
        fr->Angle = mag;
    }
    return fr;
}

//Return a 3x3 rotation matrix (expressed in radians) from a set of Euler angles.
gsl_matrix* ea_to_rotation_matrix(double* ea) {

    gsl_matrix* m = gsl_matrix_alloc(3, 3);

    double EAx = ea[0];
    double EAy = ea[1];
    double EAz = ea[2];

    double cosX = cos(EAx);
    double sinX = sin(EAx);
    double cosY = cos(EAy);
    double sinY = sin(EAy);
    double cosZ = cos(EAz);
    double sinZ = sin(EAz);

    gsl_matrix_set(m, 0, 0, cosZ * cosY);
    gsl_matrix_set(m, 0, 1, cosZ * sinY * sinX - sinZ * cosX);
    gsl_matrix_set(m, 0, 2, cosZ * sinY * cosX + sinZ * sinX);
    gsl_matrix_set(m, 1, 0, sinZ * cosY);
    gsl_matrix_set(m, 1, 1, sinZ * sinY * sinX + cosZ * cosX);
    gsl_matrix_set(m, 1, 2, sinZ * sinY * cosX - cosZ * sinX);
    gsl_matrix_set(m, 2, 0, -1 * sinY);
    gsl_matrix_set(m, 2, 1, cosY * sinX);
    gsl_matrix_set(m, 2, 2, cosY * cosX);
    
    return m;
}

//Return an array of 3x3 rotation matrices (expressed in radians) from arrays of Euler angles.
gsl_matrix** eas_to_rotation_matrices(gsl_vector* ex, gsl_vector* ey, gsl_vector* ez) {

    size_t n_size = ex->size;
    gsl_matrix** m_array = (gsl_matrix**)malloc(n_size * sizeof(gsl_matrix*));

    // Initialize each gsl_matrix in the array
    for (size_t i = 0; i < n_size; i++) {
        double* ea = (double*)malloc(3 * sizeof(double));
        ea[0] = gsl_vector_get(ex, i);
        ea[1] = gsl_vector_get(ey, i);
        ea[2] = gsl_vector_get(ez, i);
        m_array[i] = ea_to_rotation_matrix(ea);
    }

    return m_array;
}

//Return a set of Euler angles from a 3x3 rotation matrix (expressed in radians).
double* rotation_matrix_to_ea(gsl_matrix* m){
    double *euler_angles = (double*)malloc(3 * sizeof(double));

    double m32 = gsl_matrix_get(m, 2, 1);
    double m33 = gsl_matrix_get(m, 2, 1);
    double m31 = gsl_matrix_get(m, 2, 1);
    double m21 = gsl_matrix_get(m, 2, 1);
    double m11 = gsl_matrix_get(m, 2, 1);

    euler_angles[0] = atan2(m32, m33);
    euler_angles[1] = atan2(-1 * m31, pow(pow(m32, 2) + pow(m33, 2), 0.5));
    euler_angles[2] = atan2(m21, m11);

    return euler_angles;
}

//Return an array of Euler angles from an array of 3x3 rotation matrices (expressed in radians).
gsl_matrix* rotation_matrices_to_eas(gsl_matrix** m_array){
    size_t n_size = sizeof(m_array) / sizeof(m_array[0]);
    gsl_matrix* ea_array = gsl_matrix_alloc(3, n_size);
    

    for (size_t i = 0; i < n_size; i++) {
        double* ea = rotation_matrix_to_ea(m_array[i]);
        gsl_matrix_set(ea_array, 0, i, ea[0]);
        gsl_matrix_set(ea_array, 1, i, ea[1]);
        gsl_matrix_set(ea_array, 2, i, ea[2]);
    }

    return ea_array;
}

// Return the set of Euler angles from a finite rotation (expressed in degrees).
double* fr_to_euler_angles(const FiniteRot *fr_sph) {
    gsl_matrix *matrix = fr_to_rotation_matrix(fr_sph);

    double *euler_angles = (double*)malloc(3 * sizeof(double));

    euler_angles[0] = atan2(gsl_matrix_get(matrix, 2, 1), gsl_matrix_get(matrix, 2, 2));
    euler_angles[1] = atan2(-1.0 * gsl_matrix_get(matrix, 2, 0), sqrt(pow(gsl_matrix_get(matrix, 2, 1), 2) + pow(gsl_matrix_get(matrix, 2, 2), 2)));
    euler_angles[2] = atan2(gsl_matrix_get(matrix, 1, 0), gsl_matrix_get(matrix, 0, 0));

    return euler_angles;
}

// Return a finite rotation (expressed in degrees) from a set of Euler angles.
FiniteRot* ea_to_finrot(double* ea) {
    gsl_matrix* m = ea_to_rotation_matrix(ea); 
    return rotation_matrix_to_fr(m);
}