#include "../pk_structs.h"

static bool surface_velocity_is_number(PyObject *value) {
    return PyFloat_Check(value) || PyLong_Check(value);
}

static PyObject *surface_velocity_new_stat(double mean, double stdev) {
    Stat *stat = PyObject_New(Stat, &StatType);
    if (stat == NULL) {
        return NULL;
    }
    stat->Mean = mean;
    stat->StDev = stdev;
    return (PyObject *)stat;
}

static int surface_velocity_parse_numeric(PyObject *value, double *out, const char *field_name) {
    if (!surface_velocity_is_number(value)) {
        PyErr_Format(PyExc_TypeError, "%s argument must be a float", field_name);
        return -1;
    }
    *out = PyFloat_AsDouble(value);
    return 0;
}

static int surface_velocity_coerce_value(PyObject *value, PyObject **out, const char *field_name) {
    if (value == NULL || value == Py_None) {
        Py_INCREF(Py_None);
        *out = Py_None;
        return 0;
    }

    if (surface_velocity_is_number(value)) {
        *out = PyFloat_FromDouble(PyFloat_AsDouble(value));
        return *out == NULL ? -1 : 0;
    }

    if (PyObject_TypeCheck(value, &StatType)) {
        Py_INCREF(value);
        *out = value;
        return 0;
    }

    double *values = parse_double_array(value);
    if (values == NULL) {
        PyErr_Clear();
        PyErr_Format(PyExc_TypeError, "%s must be a float, Stat, None, or a 2-value array-like input", field_name);
        return -1;
    }

    if ((int)values[0] != 2) {
        free(values);
        PyErr_Format(PyExc_ValueError, "%s array-like input must contain exactly 2 values", field_name);
        return -1;
    }

    *out = surface_velocity_new_stat(values[1], values[2]);
    free(values);
    return *out == NULL ? -1 : 0;
}

static int surface_velocity_set_optional(PyObject **target, PyObject *value, const char *field_name) {
    PyObject *coerced = NULL;
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
        return -1;
    }
    if (surface_velocity_coerce_value(value, &coerced, field_name) != 0) {
        return -1;
    }
    Py_XDECREF(*target);
    *target = coerced;
    return 0;
}

static PyObject *surface_velocity_optional_repr(PyObject *value) {
    if (value == Py_None) {
        return PyUnicode_FromString("None");
    }
    if (PyObject_TypeCheck(value, &StatType)) {
        Stat *stat = (Stat *)value;
        return PyUnicode_FromFormat("%.6g +- %.6g", stat->Mean, stat->StDev);
    }
    return PyObject_Str(value);
}

static PyObject* SurfaceVelocity_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    Py_ssize_t n_args;
    PyObject *lon_obj = NULL, *lat_obj = NULL;
    PyObject *field1 = NULL, *field2 = NULL, *field3 = NULL, *field4 = NULL;
    double lon = 0.0, lat = 0.0;

    if (kwds != NULL && PyDict_Size(kwds) > 0) {
        PyErr_SetString(PyExc_TypeError, "SurfaceVelocity() does not accept keyword arguments");
        return NULL;
    }

    n_args = args == NULL ? 0 : PyTuple_Size(args);
    if (n_args == 0) {
        lon = 0.0;
        lat = 0.0;
    } else if (n_args >= 3 && n_args <= 6) {
        if (!PyArg_ParseTuple(args, "OO|OOOO", &lon_obj, &lat_obj, &field1, &field2, &field3, &field4)) {
            PyErr_SetString(PyExc_TypeError, "SurfaceVelocity() failed to parse one or more input arguments");
            return NULL;
        }
        if (surface_velocity_parse_numeric(lon_obj, &lon, "Lon") != 0 ||
            surface_velocity_parse_numeric(lat_obj, &lat, "Lat") != 0) {
            return NULL;
        }
    } else {
        PyErr_SetString(PyExc_TypeError, "SurfaceVelocity() constructor accepts either no arguments, (Lon, Lat, TotalVel), (Lon, Lat, EastVel, NorthVel), (Lon, Lat, EastVel, NorthVel, TotalVel), or (Lon, Lat, EastVel, NorthVel, TotalVel, Azimuth)");
        return NULL;
    }

    SurfaceVelocity *self = (SurfaceVelocity *)type->tp_alloc(type, 0);
    if (self == NULL) {
        return NULL;
    }

    self->Lon = lon;
    self->Lat = lat;
    Py_INCREF(Py_None);
    self->EastVel = Py_None;
    Py_INCREF(Py_None);
    self->NorthVel = Py_None;
    Py_INCREF(Py_None);
    self->TotalVel = Py_None;
    Py_INCREF(Py_None);
    self->Azimuth = Py_None;

    if (n_args == 3) {
        if (surface_velocity_set_optional(&self->TotalVel, field1, "TotalVel") != 0) {
            Py_DECREF((PyObject *)self);
            return NULL;
        }
    } else if (n_args == 4) {
        if (surface_velocity_set_optional(&self->EastVel, field1, "EastVel") != 0 ||
            surface_velocity_set_optional(&self->NorthVel, field2, "NorthVel") != 0) {
            Py_DECREF((PyObject *)self);
            return NULL;
        }
    } else if (n_args == 5) {
        if (surface_velocity_set_optional(&self->EastVel, field1, "EastVel") != 0 ||
            surface_velocity_set_optional(&self->NorthVel, field2, "NorthVel") != 0 ||
            surface_velocity_set_optional(&self->TotalVel, field3, "TotalVel") != 0) {
            Py_DECREF((PyObject *)self);
            return NULL;
        }
    } else if (n_args == 6) {
        if (surface_velocity_set_optional(&self->EastVel, field1, "EastVel") != 0 ||
            surface_velocity_set_optional(&self->NorthVel, field2, "NorthVel") != 0 ||
            surface_velocity_set_optional(&self->TotalVel, field3, "TotalVel") != 0 ||
            surface_velocity_set_optional(&self->Azimuth, field4, "Azimuth") != 0) {
            Py_DECREF((PyObject *)self);
            return NULL;
        }
    }

    return (PyObject *)self;
}

static void SurfaceVelocity_dealloc(SurfaceVelocity *self) {
    Py_XDECREF(self->EastVel);
    Py_XDECREF(self->NorthVel);
    Py_XDECREF(self->TotalVel);
    Py_XDECREF(self->Azimuth);
    Py_TYPE(self)->tp_free(self);
}

static PyObject* SurfaceVelocity_repr(SurfaceVelocity *self) {
    PyObject *east = surface_velocity_optional_repr(self->EastVel);
    PyObject *north = surface_velocity_optional_repr(self->NorthVel);
    PyObject *total = surface_velocity_optional_repr(self->TotalVel);
    PyObject *azimuth = surface_velocity_optional_repr(self->Azimuth);
    PyObject *result;

    if (east == NULL || north == NULL || total == NULL || azimuth == NULL) {
        Py_XDECREF(east);
        Py_XDECREF(north);
        Py_XDECREF(total);
        Py_XDECREF(azimuth);
        return NULL;
    }

    result = PyUnicode_FromFormat(
        "SurfaceVelocity(Lon=%.6g, Lat=%.6g, EastVel=%U, NorthVel=%U, TotalVel=%U, Azimuth=%U)",
        self->Lon, self->Lat, east, north, total, azimuth);

    Py_DECREF(east);
    Py_DECREF(north);
    Py_DECREF(total);
    Py_DECREF(azimuth);
    return result;
}

static PyObject* SurfaceVelocity_get_Lon(SurfaceVelocity *self, void *closure) {
    return PyFloat_FromDouble(self->Lon);
}

static PyObject* SurfaceVelocity_get_Lat(SurfaceVelocity *self, void *closure) {
    return PyFloat_FromDouble(self->Lat);
}

static PyObject* SurfaceVelocity_get_EastVel(SurfaceVelocity *self, void *closure) {
    Py_INCREF(self->EastVel);
    return self->EastVel;
}

static PyObject* SurfaceVelocity_get_NorthVel(SurfaceVelocity *self, void *closure) {
    Py_INCREF(self->NorthVel);
    return self->NorthVel;
}

static PyObject* SurfaceVelocity_get_TotalVel(SurfaceVelocity *self, void *closure) {
    Py_INCREF(self->TotalVel);
    return self->TotalVel;
}

static PyObject* SurfaceVelocity_get_Azimuth(SurfaceVelocity *self, void *closure) {
    Py_INCREF(self->Azimuth);
    return self->Azimuth;
}

static int SurfaceVelocity_set_Lon(SurfaceVelocity *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
        return -1;
    }
    if (!surface_velocity_is_number(value)) {
        PyErr_SetString(PyExc_TypeError, "Lon attribute must be a float");
        return -1;
    }
    self->Lon = PyFloat_AsDouble(value);
    return 0;
}

static int SurfaceVelocity_set_Lat(SurfaceVelocity *self, PyObject *value, void *closure) {
    if (value == NULL) {
        PyErr_SetString(PyExc_TypeError, "Cannot delete the attribute");
        return -1;
    }
    if (!surface_velocity_is_number(value)) {
        PyErr_SetString(PyExc_TypeError, "Lat attribute must be a float");
        return -1;
    }
    self->Lat = PyFloat_AsDouble(value);
    return 0;
}

static int SurfaceVelocity_set_EastVel(SurfaceVelocity *self, PyObject *value, void *closure) {
    return surface_velocity_set_optional(&self->EastVel, value, "EastVel");
}

static int SurfaceVelocity_set_NorthVel(SurfaceVelocity *self, PyObject *value, void *closure) {
    return surface_velocity_set_optional(&self->NorthVel, value, "NorthVel");
}

static int SurfaceVelocity_set_TotalVel(SurfaceVelocity *self, PyObject *value, void *closure) {
    return surface_velocity_set_optional(&self->TotalVel, value, "TotalVel");
}

static int SurfaceVelocity_set_Azimuth(SurfaceVelocity *self, PyObject *value, void *closure) {
    return surface_velocity_set_optional(&self->Azimuth, value, "Azimuth");
}

static PyGetSetDef SurfaceVelocity_getsetters[] = {
    {"Lon", (getter)SurfaceVelocity_get_Lon, (setter)SurfaceVelocity_set_Lon, "Longitude of the surface point in degrees-East.", NULL},
    {"Lat", (getter)SurfaceVelocity_get_Lat, (setter)SurfaceVelocity_set_Lat, "Latitude of the surface point in degrees-North.", NULL},
    {"EastVel", (getter)SurfaceVelocity_get_EastVel, (setter)SurfaceVelocity_set_EastVel, "East-component of the velocity.", NULL},
    {"NorthVel", (getter)SurfaceVelocity_get_NorthVel, (setter)SurfaceVelocity_set_NorthVel, "North-component of the velocity.", NULL},
    {"TotalVel", (getter)SurfaceVelocity_get_TotalVel, (setter)SurfaceVelocity_set_TotalVel, "Total velocity.", NULL},
    {"Azimuth", (getter)SurfaceVelocity_get_Azimuth, (setter)SurfaceVelocity_set_Azimuth, "Azimuth measured clockwise from North.", NULL},
    {NULL}
};

PyTypeObject SurfaceVelocityType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "pk_structs.SurfaceVelocity",
    .tp_doc = "SurfaceVelocity(Lon, Lat, TotalVel) or SurfaceVelocity(Lon, Lat, EastVel, NorthVel[, TotalVel[, Azimuth]])\n\nSurface velocity vector container. Optional velocity fields accept float, Stat, None, or a 2-value array-like input interpreted as [Mean, StDev].",
    .tp_basicsize = sizeof(SurfaceVelocity),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = SurfaceVelocity_new,
    .tp_dealloc = (destructor)SurfaceVelocity_dealloc,
    .tp_repr = (reprfunc)SurfaceVelocity_repr,
    .tp_getset = SurfaceVelocity_getsetters,
};