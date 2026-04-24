#define PY_ARRAY_UNIQUE_SYMBOL PLATEKIN_ARRAY_API
#include "pk_structs.h"

PyObject* py_fr_average(PyObject *self, PyObject *args);
PyObject *py_ev_average(PyObject *self, PyObject *args);
PyObject *py_calculate_surface_velocity(PyObject *self, PyObject *args);
PyObject *py_calculate_mean_surface_velocity(PyObject *self, PyObject *args);


/* Methods contained in the module */
static PyMethodDef methodsMethods[] = {
    {"average_fr", py_fr_average, METH_VARARGS, "average_fr(ensemble, time=0.0) -> FiniteRotation\n\nReturn the average finite rotation from either a list of\nFiniteRotation objects or a NumPy array of rotation matrices\nwith shape (n_size, 3, 3). The optional time argument sets the\nTime value on the returned object."},
    {"average_ev", py_ev_average, METH_VARARGS, "average_ev(ensemble, time_range=(0.0, 0.0)) -> EulerVector\n\nReturn the average Euler vector from either a NumPy array of\nCartesian coordinates with shape (3, n_size) or a list of\nEulerVector samples. The optional time_range is stored on the\nreturned EulerVector."},
    {"calculate_surface_velocity", py_calculate_surface_velocity, METH_VARARGS,
        "calculate_surface_velocity(ev_ensemble, lon, lat) -> tuple\n\n"
        "Compute per-sample surface velocities at a given geodetic point from an\n"
        "ensemble of EulerVector samples.\n\n"
        "Parameters\n----------\n"
        "ev_ensemble : list of EulerVector\n"
        "    Ensemble of Euler vector samples (Lon deg-E, Lat deg-N, AngVelocity deg/Myr).\n"
        "lon : float\n    Longitude of the surface point in degrees-East.\n"
        "lat : float\n    Latitude of the surface point in degrees-North.\n\n"
        "Returns\n-------\n"
        "tuple of four 1-D NumPy arrays (float64)\n"
        "    (east_vel, north_vel, total_vel) in cm/yr\n"
        "    (azimuth) in degrees clockwise from North."
    },
    {"calculate_mean_surface_velocity", py_calculate_mean_surface_velocity, METH_VARARGS,
        "calculate_mean_surface_velocity(ev_ensemble, lon, lat) -> SurfaceVelocity\n\n"
        "Compute mean and standard deviation of surface velocity components at a\n"
        "geodetic point from an ensemble of EulerVector samples.\n\n"
        "Parameters\n----------\n"
        "ev_ensemble : list of EulerVector\n"
        "    Ensemble of Euler vector samples (Lon deg-E, Lat deg-N, AngVelocity deg/Myr).\n"
        "lon : float\n    Longitude of the surface point in degrees-East.\n"
        "lat : float\n    Latitude of the surface point in degrees-North.\n\n"
        "Returns\n-------\n"
        "SurfaceVelocity\n"
        "    EastVel, NorthVel, TotalVel, Azimuth fields are Stat objects\n"
        "    (Mean +/- StDev) across the ensemble."
    },
  {NULL, NULL, 0, NULL}
};

static PyModuleDef pk_structs = {
    PyModuleDef_HEAD_INIT,
    .m_name = "pk_structs",
    .m_doc = "Plate kinematics structures and ensemble utilities.\n\nExports Covariance, Stat, FiniteRotation, EulerVector, SurfaceVelocity,\naverage_fr, average_ev, calculate_surface_velocity, and calculate_mean_surface_velocity.",
    .m_size = -1,
    .m_methods = methodsMethods,
};


PyMODINIT_FUNC PyInit_pk_structs(void) {
    PyObject *m;
    import_array(); // Initialize NumPy API

    if (PyType_Ready(&CovarianceType) < 0)
        return NULL;

    if (PyType_Ready(&StatType) < 0)
        return NULL;

    if (PyType_Ready(&FiniteRotationType) < 0)
        return NULL;

    if (PyType_Ready(&EulerVectorType) < 0)
        return NULL;

    if (PyType_Ready(&SurfaceVelocityType) < 0)
        return NULL;

    m = PyModule_Create(&pk_structs);
    if (m == NULL)
        return NULL;

    Py_INCREF(&CovarianceType);
    PyModule_AddObject(m, "Covariance", (PyObject *)&CovarianceType);

    Py_INCREF(&StatType);
    PyModule_AddObject(m, "Stat", (PyObject *)&StatType);

    Py_INCREF(&FiniteRotationType);
    PyModule_AddObject(m, "FiniteRotation", (PyObject *)&FiniteRotationType);

    Py_INCREF(&EulerVectorType);
    PyModule_AddObject(m, "EulerVector", (PyObject *)&EulerVectorType);

    Py_INCREF(&SurfaceVelocityType);
    PyModule_AddObject(m, "SurfaceVelocity", (PyObject *)&SurfaceVelocityType);
    return m;
}