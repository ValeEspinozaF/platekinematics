#define PY_ARRAY_UNIQUE_SYMBOL PLATEKIN_ARRAY_API // Must be defined before importing numpy/arrayobject.h
#define NO_IMPORT_ARRAY
/*
 * surface_velocity.c
 *
 * Implements surface velocity calculations given an ensemble of EulerVector 
 * samples and a surface point, compute the per-sample velocity components 
 * (East, North, Total, Azimuth) and, for the * mean version, return a 
 * SurfaceVelocity object with Stat fields.
 *
 */

#include "ensemble_methods.h"


/* ---------------------------------------------------------------------------
 * Internal helpers
 * ---------------------------------------------------------------------------*/

/*
 * geographical_coords_to_cartesian
 *
 * Converts a geodetic lon/lat (degrees, WGS-84 ellipsoid, height = 0) to
 * Earth-centred Cartesian coordinates in metres.
 */
static void geographical_coords_to_cartesian(double lon_deg, double lat_deg,
                                             double *x, double *y, double *z)
{
    /* WGS-84 parameters */
    const double a   = 6378137.0;
    const double f   = 1.0 / 298.257223563;
    const double b   = a * (1.0 - f);
    const double ecc = 2.0 * f - f * f;   /* e² */

    double lon_rad = to_radians(lon_deg);
    double lat_rad = to_radians(lat_deg);

    double sin_lat = sin(lat_rad);
    double cos_lat = cos(lat_rad);
    double sin_lon = sin(lon_rad);
    double cos_lon = cos(lon_rad);

    /* Radius of curvature in the prime vertical */
    double N = a / sqrt(1.0 - sin_lat * sin_lat * ecc);

    *x = cos_lat * cos_lon * N;
    *y = cos_lat * sin_lon * N;
    *z = sin_lat * (N * (b * b / (a * a)));
}


/*
 * cartesian_velocity_to_en
 *
 * Rotates an array of Cartesian velocity vectors (vX, vY, vZ) into local
 * East (vE) and North (vN) components using the standard ECEF → ENU
 * rotation matrix evaluated at the given surface point.
 */
static void cartesian_velocity_to_en(double lon_deg, double lat_deg,
                                     const double *vX, const double *vY,
                                     const double *vZ, int n_size,
                                     double *eastVel, double *northVel)
{
    double lon_rad = to_radians(lon_deg);
    double lat_rad = to_radians(lat_deg);

    double sin_lat = sin(lat_rad);
    double cos_lat = cos(lat_rad);
    double sin_lon = sin(lon_rad);
    double cos_lon = cos(lon_rad);

    /* Row 0 → North component */
    double R11 = -sin_lat * cos_lon;
    double R12 = -sin_lat * sin_lon;
    double R13 =  cos_lat;
    /* Row 1 → East component */
    double R21 = -sin_lon;
    double R22 =  cos_lon;
    /* R23 = 0 */

    for (int i = 0; i < n_size; i++) {
        northVel[i] = R11 * vX[i] + R12 * vY[i] + R13 * vZ[i];
        eastVel[i]  = R21 * vX[i] + R22 * vY[i];  /* R23 * vZ = 0 */
    }
}


/*
 * array_mean / array_std
 *
 * Simple descriptive statistics helpers
 */
static double array_mean(const double *arr, int n)
{
    double s = 0.0;
    for (int i = 0; i < n; i++) s += arr[i];
    return s / n;
}

static double array_std(const double *arr, int n, double mean)
{
    if (n <= 1) return 0.0;
    double s = 0.0;
    for (int i = 0; i < n; i++) {
        double d = arr[i] - mean;
        s += d * d;
    }
    return sqrt(s / (n - 1));
}


/*
 * make_stat
 *
 * Allocate a new Stat Python object with the given mean and stdev.
 * Returns a new reference; caller owns it.
 */
static PyObject *make_stat(double mean, double stdev)
{
    Stat *st = PyObject_New(Stat, &StatType);
    if (st == NULL) return NULL;
    st->Mean  = mean;
    st->StDev = stdev;
    return (PyObject *)st;
}


/* ---------------------------------------------------------------------------
 * Core C computation
 *
 * Accepts pre-extracted Euler-vector spherical components as three parallel
 * arrays (lon[], lat[], ang_vel[], all in degrees / deg/Myr) plus the target
 * point and writes the four velocity arrays into caller-provided buffers.
 * ---------------------------------------------------------------------------*/
static void compute_surface_velocities(
    const double *ev_lon, const double *ev_lat, const double *ev_ang,
    int n_size,
    double pnt_lon, double pnt_lat,
    double *eastVel, double *northVel,
    double *totalVel, double *azimuth)
{
    /* Point in ECEF metres */
    double px, py, pz;
    geographical_coords_to_cartesian(pnt_lon, pnt_lat, &px, &py, &pz);

    /* Cartesian velocity components (m/Myr) for each sample */
    double *vcX = (double *)malloc(n_size * sizeof(double));
    double *vcY = (double *)malloc(n_size * sizeof(double));
    double *vcZ = (double *)malloc(n_size * sizeof(double));

    for (int i = 0; i < n_size; i++) {
        double ev_cart[3];
        sph2cart(ev_lon[i], ev_lat[i], ev_ang[i], ev_cart);

        /* Convert deg/Myr Cartesian components → rad/Myr */
        double wx = to_radians(ev_cart[0]);
        double wy = to_radians(ev_cart[1]);
        double wz = to_radians(ev_cart[2]);

        /* Cross product ω × r  →  m/Myr, scaled to cm/yr */
        double scale = 1e2 / 1e6;   /* m/Myr → cm/yr */
        vcX[i] = (wy * pz - wz * py) * scale;
        vcY[i] = (wz * px - wx * pz) * scale;
        vcZ[i] = (wx * py - wy * px) * scale;
    }

    /* Rotate Cartesian velocity → East / North components (cm/yr) */
    cartesian_velocity_to_en(pnt_lon, pnt_lat,
                             vcX, vcY, vcZ, n_size,
                             eastVel, northVel);

    free(vcX);
    free(vcY);
    free(vcZ);

    /* Total speed and azimuth */
    for (int i = 0; i < n_size; i++) {
        totalVel[i] = sqrt(eastVel[i] * eastVel[i] + northVel[i] * northVel[i]);

        double at = to_degrees(atan(northVel[i] / eastVel[i]));
        double abs_at = fabs(at);
        double sign_n = (northVel[i] > 0.0) ? 1.0 : (northVel[i] < 0.0) ? -1.0 : 0.0;

        if (eastVel[i] > 0.0) {
            azimuth[i] = 90.0 - sign_n * abs_at;
        } else {
            azimuth[i] = -90.0 + sign_n * abs_at;
        }
    }
}


/* ---------------------------------------------------------------------------
 * Helper: extract EulerVector list → parallel double arrays
 * ---------------------------------------------------------------------------*/
static int pylist_ev_extract(PyObject *ev_list,
                             double **out_lon, double **out_lat,
                             double **out_ang, int *out_n)
{
    if (!PyList_Check(ev_list)) {
        PyErr_SetString(PyExc_TypeError,
            "Ensemble must be a list of EulerVector objects");
        return -1;
    }

    int n = (int)PyList_Size(ev_list);
    if (n == 0) {
        PyErr_SetString(PyExc_ValueError, "EulerVector ensemble is empty");
        return -1;
    }

    double *lon = (double *)malloc(n * sizeof(double));
    double *lat = (double *)malloc(n * sizeof(double));
    double *ang = (double *)malloc(n * sizeof(double));

    if (!lon || !lat || !ang) {
        free(lon); free(lat); free(ang);
        PyErr_NoMemory();
        return -1;
    }

    for (int i = 0; i < n; i++) {
        PyObject *item = PyList_GetItem(ev_list, i);   /* borrowed ref */
        if (!PyObject_TypeCheck(item, &EulerVectorType)) {
            free(lon); free(lat); free(ang);
            PyErr_Format(PyExc_TypeError,
                "Item %d in ensemble is not an EulerVector instance", i);
            return -1;
        }
        EulerVector *ev = (EulerVector *)item;
        lon[i] = ev->Lon;
        lat[i] = ev->Lat;
        ang[i] = ev->AngVelocity;
    }

    *out_lon = lon;
    *out_lat = lat;
    *out_ang = ang;
    *out_n   = n;
    return 0;
}


/* ---------------------------------------------------------------------------
 * Python-exposed functions
 * ---------------------------------------------------------------------------*/

/*
 * calculate_surface_velocity(ev_ensemble, lon, lat) -> tuple
 *
 * Parameters
 * ----------
 * ev_ensemble : list of EulerVector
 *     Ensemble of Euler vector samples in spherical coordinates
 *     (Lon deg-E, Lat deg-N, AngVelocity deg/Myr).
 * lon : float
 *     Longitude of the surface point in degrees-East.
 * lat : float
 *     Latitude of the surface point in degrees-North.
 *
 * Returns
 * -------
 * tuple of four 1-D NumPy arrays (float64)
 *     (east_vel, north_vel, total_vel, azimuth), all in cm/yr except
 *     azimuth which is in degrees clockwise from North.
 */
PyObject *py_calculate_surface_velocity(PyObject *self, PyObject *args)
{
    PyObject *ev_pyob;
    double pnt_lon, pnt_lat;

    if (!PyArg_ParseTuple(args, "Odd", &ev_pyob, &pnt_lon, &pnt_lat)) {
        PyErr_SetString(PyExc_TypeError,
            "calculate_surface_velocity(ev_ensemble, lon, lat)");
        return NULL;
    }

    double *ev_lon = NULL, *ev_lat = NULL, *ev_ang = NULL;
    int n_size = 0;

    if (pylist_ev_extract(ev_pyob, &ev_lon, &ev_lat, &ev_ang, &n_size) < 0)
        return NULL;

    double *eastVel  = (double *)malloc(n_size * sizeof(double));
    double *northVel = (double *)malloc(n_size * sizeof(double));
    double *totalVel = (double *)malloc(n_size * sizeof(double));
    double *azimuth  = (double *)malloc(n_size * sizeof(double));

    if (!eastVel || !northVel || !totalVel || !azimuth) {
        free(ev_lon); free(ev_lat); free(ev_ang);
        free(eastVel); free(northVel); free(totalVel); free(azimuth);
        PyErr_NoMemory();
        return NULL;
    }

    compute_surface_velocities(ev_lon, ev_lat, ev_ang, n_size,
                               pnt_lon, pnt_lat,
                               eastVel, northVel, totalVel, azimuth);

    free(ev_lon); free(ev_lat); free(ev_ang);

    /* Wrap results as NumPy arrays (copies data; C buffers are freed) */
    npy_intp dims[1] = {n_size};
    PyObject *arr_east  = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, eastVel);
    PyObject *arr_north = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, northVel);
    PyObject *arr_total = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, totalVel);
    PyObject *arr_az    = PyArray_SimpleNewFromData(1, dims, NPY_DOUBLE, azimuth);

    if (!arr_east || !arr_north || !arr_total || !arr_az) {
        Py_XDECREF(arr_east); Py_XDECREF(arr_north);
        Py_XDECREF(arr_total); Py_XDECREF(arr_az);
        free(eastVel); free(northVel); free(totalVel); free(azimuth);
        return NULL;
    }

    /* Transfer buffer ownership to NumPy via a PyCapsule so they are
       freed when the arrays are garbage-collected. */
    PyObject *cap_east  = PyCapsule_New(eastVel,  NULL, (PyCapsule_Destructor)free);
    PyObject *cap_north = PyCapsule_New(northVel, NULL, (PyCapsule_Destructor)free);
    PyObject *cap_total = PyCapsule_New(totalVel, NULL, (PyCapsule_Destructor)free);
    PyObject *cap_az    = PyCapsule_New(azimuth,  NULL, (PyCapsule_Destructor)free);

    if (cap_east)  PyArray_SetBaseObject((PyArrayObject *)arr_east,  cap_east);
    if (cap_north) PyArray_SetBaseObject((PyArrayObject *)arr_north, cap_north);
    if (cap_total) PyArray_SetBaseObject((PyArrayObject *)arr_total, cap_total);
    if (cap_az)    PyArray_SetBaseObject((PyArrayObject *)arr_az,    cap_az);

    PyObject *result = PyTuple_Pack(4, arr_east, arr_north, arr_total, arr_az);
    Py_DECREF(arr_east); Py_DECREF(arr_north);
    Py_DECREF(arr_total); Py_DECREF(arr_az);
    return result;
}


/*
 * calculate_mean_surface_velocity(ev_ensemble, lon, lat) -> SurfaceVelocity
 *
 * Parameters
 * ----------
 * ev_ensemble : list of EulerVector
 *     Ensemble of Euler vector samples.
 * lon : float
 *     Longitude of the surface point in degrees-East.
 * lat : float
 *     Latitude of the surface point in degrees-North.
 *
 * Returns
 * -------
 * SurfaceVelocity
 *     Object whose EastVel, NorthVel, TotalVel, and Azimuth fields are
 *     Stat instances containing the sample mean and standard deviation
 *     of the corresponding velocity component across the ensemble.
 */
PyObject *py_calculate_mean_surface_velocity(PyObject *self, PyObject *args)
{
    PyObject *ev_pyob;
    double pnt_lon, pnt_lat;

    if (!PyArg_ParseTuple(args, "Odd", &ev_pyob, &pnt_lon, &pnt_lat)) {
        PyErr_SetString(PyExc_TypeError,
            "calculate_mean_surface_velocity(ev_ensemble, lon, lat)");
        return NULL;
    }

    double *ev_lon = NULL, *ev_lat = NULL, *ev_ang = NULL;
    int n_size = 0;

    if (pylist_ev_extract(ev_pyob, &ev_lon, &ev_lat, &ev_ang, &n_size) < 0)
        return NULL;

    double *eastVel  = (double *)malloc(n_size * sizeof(double));
    double *northVel = (double *)malloc(n_size * sizeof(double));
    double *totalVel = (double *)malloc(n_size * sizeof(double));
    double *azimuth  = (double *)malloc(n_size * sizeof(double));

    if (!eastVel || !northVel || !totalVel || !azimuth) {
        free(ev_lon); free(ev_lat); free(ev_ang);
        free(eastVel); free(northVel); free(totalVel); free(azimuth);
        PyErr_NoMemory();
        return NULL;
    }

    compute_surface_velocities(ev_lon, ev_lat, ev_ang, n_size,
                               pnt_lon, pnt_lat,
                               eastVel, northVel, totalVel, azimuth);

    free(ev_lon); free(ev_lat); free(ev_ang);

    /* Compute statistics */
    double m_east  = array_mean(eastVel,  n_size);
    double m_north = array_mean(northVel, n_size);
    double m_total = array_mean(totalVel, n_size);
    double m_az    = array_mean(azimuth,  n_size);

    double s_east  = array_std(eastVel,  n_size, m_east);
    double s_north = array_std(northVel, n_size, m_north);
    double s_total = array_std(totalVel, n_size, m_total);
    double s_az    = array_std(azimuth,  n_size, m_az);

    free(eastVel); free(northVel); free(totalVel); free(azimuth);

    /* Build Stat objects */
    PyObject *stat_east  = make_stat(m_east,  s_east);
    PyObject *stat_north = make_stat(m_north, s_north);
    PyObject *stat_total = make_stat(m_total, s_total);
    PyObject *stat_az    = make_stat(m_az,    s_az);

    if (!stat_east || !stat_north || !stat_total || !stat_az) {
        Py_XDECREF(stat_east); Py_XDECREF(stat_north);
        Py_XDECREF(stat_total); Py_XDECREF(stat_az);
        return NULL;
    }

    /* Build SurfaceVelocity object */
    SurfaceVelocity *sv = PyObject_New(SurfaceVelocity, &SurfaceVelocityType);
    if (sv == NULL) {
        Py_DECREF(stat_east); Py_DECREF(stat_north);
        Py_DECREF(stat_total); Py_DECREF(stat_az);
        return NULL;
    }

    sv->Lon      = pnt_lon;
    sv->Lat      = pnt_lat;
    sv->EastVel  = stat_east;   /* SurfaceVelocity owns these refs */
    sv->NorthVel = stat_north;
    sv->TotalVel = stat_total;
    sv->Azimuth  = stat_az;

    return (PyObject *)sv;
}
