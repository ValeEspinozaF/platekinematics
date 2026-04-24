# platekinematics
Python extension module with tools for easy handling of plate kinematic functions.


# requirements
numpy = 1.20.1
gsl = 2.7.1


# build
The extension modules are built from `setup.py`.

For a local rebuild on Windows from the repository root:

```bash
python -m pip install -U pip setuptools wheel numpy
python setup.py build_ext --inplace
```

If GSL is not installed in the bundled local vcpkg path, point the build at a
different GSL library directory first:

```bash
set PK_GSL_LIB_DIR=path\to\gsl\lib
python setup.py build_ext --inplace
```

In this repository, the default local path expected by `setup.py` is:

```text
src\vcpkg\installed\x64-windows\lib
```

The corresponding runtime DLLs are expected under:

```text
src\vcpkg\installed\x64-windows\bin
```


# quick api reference
`platekinematics.pk_structs` exports these main objects and functions:

`Covariance(values=None)`
- `values` must contain 6 floats ordered as `[C11, C12, C13, C22, C23, C33]`
- `to_numpy()` returns shape `(6,)`

`Stat(mean, stdev)` or `Stat([mean, stdev])`
- stores `Mean` and `StDev`
- useful for uncertain velocity components

`FiniteRotation(Lon, Lat, Angle, Time, Covariance=None)`
- `to_numpy()` returns shape `(4,)` without covariance or `(10,)` with covariance
- `build_array(n_size)` returns sampled rotation matrices with shape `(n_size, 3, 3)`
- `build_ensemble(n_size)` returns a Python list of sampled `FiniteRotation` objects

`EulerVector(Lon, Lat, AngVelocity, TimeRange, Covariance=None)`
- `TimeRange` must be a 2-item tuple of floats
- `to_numpy()` returns shape `(5,)` without covariance or `(11,)` with covariance
- `build_array(n_size, coordinate_system='cartesian')` returns shape `(3, n_size)`
- `build_ensemble(n_size)` returns a Python list of sampled `EulerVector` objects

`SurfaceVelocity(Lon, Lat, TotalVel)`
`SurfaceVelocity(Lon, Lat, EastVel, NorthVel[, TotalVel[, Azimuth]])`
- `EastVel`, `NorthVel`, `TotalVel`, and `Azimuth` accept `float`, `Stat`, `None`, or a 2-value array-like input interpreted as `[Mean, StDev]`
- `Lon` and `Lat` are stored in degrees-East and degrees-North

`average_fr(ensemble, time=0.0)`
- accepts either a list of `FiniteRotation` objects or a NumPy array with shape `(n_size, 3, 3)`
- returns one `FiniteRotation`

`average_ev(ensemble, time_range=(0.0, 0.0))`
- accepts either a NumPy array with shape `(3, n_size)` or a list of `EulerVector` objects
- returns one `EulerVector`

`platekinematics.spherical_functions` also provides:

- `to_degrees(angle_radians)`
- `to_radians(angle_degrees)`
- `sph2cart(lon, lat, magnitude)`
- `cart2sph(x, y, z)`


# quick check
After rebuilding, a minimal import check is:

```bash
python -c "import platekinematics; from platekinematics.pk_structs import Covariance, FiniteRotation, EulerVector"
```