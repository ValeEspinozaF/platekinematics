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


static int covariance_is_zero(const EulerVector *ev)
{
    if (!ev->has_covariance)
        return 1;

    return ev->Covariance.C11 == 0.0 &&
           ev->Covariance.C12 == 0.0 &&
           ev->Covariance.C13 == 0.0 &&
           ev->Covariance.C22 == 0.0 &&
           ev->Covariance.C23 == 0.0 &&
           ev->Covariance.C33 == 0.0;
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


/* Forward declaration: defined in eulervec_methods.c */
PyObject *build_ev_ensemble(EulerVector *ev_sph, int n_size);


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

/* ---------------------------------------------------------------------------
 * calculate_sv_from_single_ev
 *
 * Internal helper for the single-EulerVector + multi-point overload.
 *
 * If ev->has_covariance: builds an ensemble of ens_n_size samples and
 * returns a list of SurfaceVelocity objects with Stat fields.
 * Otherwise uses the single EV directly and returns a list with plain
 * float velocity fields.
 * ---------------------------------------------------------------------------*/
static PyObject *calculate_sv_from_single_ev(
    EulerVector *ev, PyObject *lon_seq, PyObject *lat_seq, int ens_n_size)
{
    /* ----- parse lon/lat point sequences ---------------------------------- */
    if (!PySequence_Check(lon_seq) || !PySequence_Check(lat_seq)) {
        PyErr_SetString(PyExc_TypeError,
            "lon and lat arguments must be sequences (list or tuple)");
        return NULL;
    }
    int n_points = (int)PySequence_Length(lon_seq);
    if (n_points != (int)PySequence_Length(lat_seq)) {
        PyErr_SetString(PyExc_ValueError, "lon and lat sequences must have equal length");
        return NULL;
    }
    if (n_points == 0) {
        PyErr_SetString(PyExc_ValueError, "lon / lat sequences are empty");
        return NULL;
    }

    double *pnt_lons = (double *)malloc(n_points * sizeof(double));
    double *pnt_lats = (double *)malloc(n_points * sizeof(double));
    if (!pnt_lons || !pnt_lats) {
        free(pnt_lons); free(pnt_lats);
        PyErr_NoMemory();
        return NULL;
    }
    for (int i = 0; i < n_points; i++) {
        PyObject *lo = PySequence_GetItem(lon_seq, i);
        PyObject *la = PySequence_GetItem(lat_seq, i);
        if (!lo || !la) {
            Py_XDECREF(lo); Py_XDECREF(la);
            free(pnt_lons); free(pnt_lats);
            return NULL;
        }
        pnt_lons[i] = PyFloat_AsDouble(lo);
        pnt_lats[i] = PyFloat_AsDouble(la);
        Py_DECREF(lo); Py_DECREF(la);
    }

    /* ----- build EV arrays ------------------------------------------------ */
    double *ev_lon = NULL, *ev_lat = NULL, *ev_ang = NULL;
    int ens_size = 0;
    int use_stat = 0;
    PyObject *ens_list = NULL;

    if (ev->has_covariance && !covariance_is_zero(ev)) {
        ens_list = build_ev_ensemble(ev, ens_n_size);
        if (ens_list == NULL) {
            free(pnt_lons); free(pnt_lats);
            return NULL;
        }
        if (pylist_ev_extract(ens_list, &ev_lon, &ev_lat, &ev_ang, &ens_size) < 0) {
            Py_DECREF(ens_list);
            free(pnt_lons); free(pnt_lats);
            return NULL;
        }
        Py_DECREF(ens_list);
        use_stat = 1;
    } else {
        ev_lon = (double *)malloc(sizeof(double));
        ev_lat = (double *)malloc(sizeof(double));
        ev_ang = (double *)malloc(sizeof(double));
        if (!ev_lon || !ev_lat || !ev_ang) {
            free(ev_lon); free(ev_lat); free(ev_ang);
            free(pnt_lons); free(pnt_lats);
            PyErr_NoMemory();
            return NULL;
        }
        ev_lon[0] = ev->Lon;
        ev_lat[0] = ev->Lat;
        ev_ang[0] = ev->AngVelocity;
        ens_size  = 1;
        use_stat  = 0;
    }

    /* ----- per-point computation ----------------------------------------- */
    PyObject *result = PyList_New(n_points);
    if (!result) {
        free(ev_lon); free(ev_lat); free(ev_ang);
        free(pnt_lons); free(pnt_lats);
        return NULL;
    }

    double *eastVel  = (double *)malloc(ens_size * sizeof(double));
    double *northVel = (double *)malloc(ens_size * sizeof(double));
    double *totalVel = (double *)malloc(ens_size * sizeof(double));
    double *azimuth  = (double *)malloc(ens_size * sizeof(double));

    if (!eastVel || !northVel || !totalVel || !azimuth) {
        free(eastVel); free(northVel); free(totalVel); free(azimuth);
        free(ev_lon); free(ev_lat); free(ev_ang);
        free(pnt_lons); free(pnt_lats);
        Py_DECREF(result);
        PyErr_NoMemory();
        return NULL;
    }

    for (int i = 0; i < n_points; i++) {
        double pnt_lon = pnt_lons[i];
        double pnt_lat = pnt_lats[i];

        compute_surface_velocities(ev_lon, ev_lat, ev_ang, ens_size,
                                   pnt_lon, pnt_lat,
                                   eastVel, northVel, totalVel, azimuth);

        double m_east  = array_mean(eastVel,  ens_size);
        double m_north = array_mean(northVel, ens_size);
        double m_total = array_mean(totalVel, ens_size);
        double m_az    = array_mean(azimuth,  ens_size);

        SurfaceVelocity *sv = PyObject_New(SurfaceVelocity, &SurfaceVelocityType);
        if (sv == NULL) {
            free(eastVel); free(northVel); free(totalVel); free(azimuth);
            free(ev_lon); free(ev_lat); free(ev_ang);
            free(pnt_lons); free(pnt_lats);
            Py_DECREF(result);
            return NULL;
        }

        sv->Lon = pnt_lon;
        sv->Lat = pnt_lat;

        if (use_stat) {
            double s_east  = array_std(eastVel,  ens_size, m_east);
            double s_north = array_std(northVel, ens_size, m_north);
            double s_total = array_std(totalVel, ens_size, m_total);
            double s_az    = array_std(azimuth,  ens_size, m_az);
            sv->EastVel  = make_stat(m_east,  s_east);
            sv->NorthVel = make_stat(m_north, s_north);
            sv->TotalVel = make_stat(m_total, s_total);
            sv->Azimuth  = make_stat(m_az,    s_az);
        } else {
            sv->EastVel  = PyFloat_FromDouble(m_east);
            sv->NorthVel = PyFloat_FromDouble(m_north);
            sv->TotalVel = PyFloat_FromDouble(m_total);
            sv->Azimuth  = PyFloat_FromDouble(m_az);
        }

        PyList_SET_ITEM(result, i, (PyObject *)sv);
    }

    free(eastVel); free(northVel); free(totalVel); free(azimuth);
    free(ev_lon); free(ev_lat); free(ev_ang);
    free(pnt_lons); free(pnt_lats);
    return result;
}


/*
 * calculate_surface_velocity — two call signatures dispatched by argument types
 *
 * Signature A  (list-of-EVs, single point):
 *   calculate_surface_velocity(ev_ensemble, lon, lat) -> tuple
 *   ev_ensemble : list of EulerVector
 *   lon, lat    : float — geodetic coordinates of the surface point
 *   Returns a tuple of four 1-D NumPy arrays
 *   (east_vel, north_vel, total_vel, azimuth), cm/yr / degrees.
 *
 * Signature B  (single EV, arrays of points):
 *   calculate_surface_velocity(ev, lons, lats [, n_size=100000]) -> list
 *   ev          : EulerVector — if it carries a Covariance an ensemble of
 *                 n_size samples is drawn; otherwise the single vector is used.
 *   lons, lats  : sequence of float — geodetic coordinates of the surface points
 *   n_size      : int (optional, default 100000) — ensemble size when covariance
 *                 is present.
 *   Returns a list of SurfaceVelocity objects, one per point.
 *   Each SurfaceVelocity has Stat fields when covariance is present, plain
 *   float fields otherwise.
 */
PyObject *py_calculate_surface_velocity(PyObject *self, PyObject *args)
{
    int n_args = (int)PyTuple_Size(args);
    if (n_args < 1) {
        PyErr_SetString(PyExc_TypeError,
            "calculate_surface_velocity() requires at least one argument");
        return NULL;
    }

    PyObject *first = PyTuple_GET_ITEM(args, 0);  /* borrowed */

    /* ----- Signature B: single EulerVector + point sequences ------------- */
    if (PyObject_TypeCheck(first, &EulerVectorType)) {
        if (n_args < 3 || n_args > 4) {
            PyErr_SetString(PyExc_TypeError,
                "calculate_surface_velocity(ev, lons, lats [, n_size=100000])");
            return NULL;
        }
        PyObject *lon_seq  = PyTuple_GET_ITEM(args, 1);
        PyObject *lat_seq  = PyTuple_GET_ITEM(args, 2);
        int ens_n_size = 100000;
        if (n_args == 4) {
            PyObject *ns = PyTuple_GET_ITEM(args, 3);
            if (!PyLong_Check(ns)) {
                PyErr_SetString(PyExc_TypeError, "n_size must be an integer");
                return NULL;
            }
            ens_n_size = (int)PyLong_AsLong(ns);
        }
        return calculate_sv_from_single_ev(
            (EulerVector *)first, lon_seq, lat_seq, ens_n_size);
    }

    /* ----- Signature A: list-of-EVs + single float point ----------------- */
    if (!PyList_Check(first)) {
        PyErr_SetString(PyExc_TypeError,
            "First argument must be a list of EulerVector objects "
            "or a single EulerVector");
        return NULL;
    }

    if (n_args != 3) {
        PyErr_SetString(PyExc_TypeError,
            "calculate_surface_velocity(ev_ensemble, lon, lat)");
        return NULL;
    }

    PyObject *lon_obj = PyTuple_GET_ITEM(args, 1);
    PyObject *lat_obj = PyTuple_GET_ITEM(args, 2);
    if (!PyFloat_Check(lon_obj) || !PyFloat_Check(lat_obj)) {
        PyErr_SetString(PyExc_TypeError, "lon and lat must be floats");
        return NULL;
    }
    double pnt_lon = PyFloat_AsDouble(lon_obj);
    double pnt_lat = PyFloat_AsDouble(lat_obj);

    double *ev_lon = NULL, *ev_lat = NULL, *ev_ang = NULL;
    int n_size = 0;
    if (pylist_ev_extract(first, &ev_lon, &ev_lat, &ev_ang, &n_size) < 0)
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

    /* Wrap results as NumPy-owned arrays and copy computed buffers. */
    npy_intp dims[1] = {n_size};
    PyObject *arr_east  = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyObject *arr_north = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyObject *arr_total = PyArray_SimpleNew(1, dims, NPY_DOUBLE);
    PyObject *arr_az    = PyArray_SimpleNew(1, dims, NPY_DOUBLE);

    if (!arr_east || !arr_north || !arr_total || !arr_az) {
        Py_XDECREF(arr_east); Py_XDECREF(arr_north);
        Py_XDECREF(arr_total); Py_XDECREF(arr_az);
        free(eastVel); free(northVel); free(totalVel); free(azimuth);
        return NULL;
    }

    double *arr_east_data  = (double *)PyArray_DATA((PyArrayObject *)arr_east);
    double *arr_north_data = (double *)PyArray_DATA((PyArrayObject *)arr_north);
    double *arr_total_data = (double *)PyArray_DATA((PyArrayObject *)arr_total);
    double *arr_az_data    = (double *)PyArray_DATA((PyArrayObject *)arr_az);

    for (int i = 0; i < n_size; i++) {
        arr_east_data[i] = eastVel[i];
        arr_north_data[i] = northVel[i];
        arr_total_data[i] = totalVel[i];
        arr_az_data[i] = azimuth[i];
    }

    free(eastVel);
    free(northVel);
    free(totalVel);
    free(azimuth);

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
