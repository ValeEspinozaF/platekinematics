import numpy as np
import pytest

from platekinematics import pk_structs as pk


ZERO_COV_VALUES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_calculate_surface_velocity_list_overload(ev):
    ev_ens = ev.build_ensemble(32)
    east, north, total, azimuth = pk.calculate_surface_velocity(ev_ens, 10.0, 45.0)

    for arr in (east, north, total, azimuth):
        assert isinstance(arr, np.ndarray)
        assert arr.shape == (32,)
        assert np.all(np.isfinite(arr))


def test_calculate_surface_velocity_single_ev_no_covariance_uses_floats():
    ev_no_cov = pk.EulerVector(45.0, 20.0, 0.5, (0.0, 0.0))
    lons = [10.0, 20.0, 30.0]
    lats = [10.0, 20.0, 30.0]

    out = pk.calculate_surface_velocity(ev_no_cov, lons, lats)
    assert isinstance(out, list)
    assert len(out) == 3
    assert all(isinstance(item, pk.SurfaceVelocity) for item in out)
    assert isinstance(out[0].EastVel, float)
    assert isinstance(out[0].NorthVel, float)
    assert isinstance(out[0].TotalVel, float)
    assert isinstance(out[0].Azimuth, float)


def test_calculate_surface_velocity_single_ev_zero_covariance_uses_floats():
    cov_zero = pk.Covariance(ZERO_COV_VALUES)
    ev_zero_cov = pk.EulerVector(45.0, 20.0, 0.5, (0.0, 0.0), cov_zero)
    out = pk.calculate_surface_velocity(ev_zero_cov, [10.0, 20.0], [10.0, 20.0], 128)
    assert isinstance(out[0].EastVel, float)


def test_calculate_surface_velocity_single_ev_with_covariance_uses_stats(cov):
    ev_cov = pk.EulerVector(45.0, 20.0, 0.5, (0.0, 0.0), cov)
    out = pk.calculate_surface_velocity(ev_cov, [10.0, 20.0], [10.0, 20.0], 256)
    assert isinstance(out, list)
    assert len(out) == 2
    assert isinstance(out[0], pk.SurfaceVelocity)
    assert isinstance(out[0].EastVel, pk.Stat)
    assert isinstance(out[0].NorthVel, pk.Stat)
    assert isinstance(out[0].TotalVel, pk.Stat)
    assert isinstance(out[0].Azimuth, pk.Stat)


def test_calculate_surface_velocity_rejects_bad_inputs(ev):
    ev_ens = ev.build_ensemble(4)

    with pytest.raises(TypeError):
        pk.calculate_surface_velocity(123, 10.0, 20.0)

    with pytest.raises(TypeError):
        pk.calculate_surface_velocity(ev_ens, "10.0", 20.0)

    with pytest.raises(ValueError):
        pk.calculate_surface_velocity(ev, [10.0, 20.0], [10.0], 100)


def test_calculate_mean_surface_velocity(ev):
    ev_ens = ev.build_ensemble(64)
    sv = pk.calculate_mean_surface_velocity(ev_ens, 10.0, 45.0)

    assert isinstance(sv, pk.SurfaceVelocity)
    assert isinstance(sv.EastVel, pk.Stat)
    assert isinstance(sv.NorthVel, pk.Stat)
    assert isinstance(sv.TotalVel, pk.Stat)
    assert isinstance(sv.Azimuth, pk.Stat)
