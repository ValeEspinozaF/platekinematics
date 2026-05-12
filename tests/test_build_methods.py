import numpy as np
import pytest

import platekinematics as pk

def _sph2cart(lon_deg, lat_deg, magnitude):
    lon = np.deg2rad(lon_deg)
    lat = np.deg2rad(lat_deg)
    x = magnitude * np.cos(lat) * np.cos(lon)
    y = magnitude * np.cos(lat) * np.sin(lon)
    z = magnitude * np.sin(lat)
    return np.array([x, y, z], dtype=float)


#-- FiniteRotation build methods
def test_finite_rotation_build_array(fr):
    mats = fr.build_array(16)
    assert isinstance(mats, np.ndarray)
    assert mats.shape == (16, 3, 3)


def test_finite_rotation_build_ensemble(fr):
    ens = fr.build_ensemble(12)
    assert isinstance(ens, list)
    assert len(ens) == 12
    assert all(isinstance(item, pk.FiniteRotation) for item in ens)


def test_finite_rotation_build_methods_fail_without_covariance():
    fr_no_cov = pk.FiniteRotation(1.0, 2.0, 3.0, 4.0)
    with pytest.raises(TypeError):
        fr_no_cov.build_array(8)
    with pytest.raises(TypeError):
        fr_no_cov.build_ensemble(8)


def test_finite_rotation_build_array_rotation_matrix_accuracy(fr):
    mats = fr.build_array(64)
    eye = np.eye(3)

    for i in range(mats.shape[0]):
        r = mats[i]
        assert np.allclose(r.T @ r, eye, atol=1e-8)
        assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-8)


def test_finite_rotation_build_array_accuracy(fr):
    expected = _sph2cart(fr.Lon, fr.Lat, fr.Angle)

    n_samples = 10000
    mats = fr.build_array(n_samples)
    rot_vecs = np.einsum("ijk,j->ik", mats, expected)
    sample_mean = np.mean(rot_vecs, axis=0)

    assert np.allclose(sample_mean, expected, atol=7e-3)


def test_finite_rotation_build_ensemble_accuracy(fr):   
    expected = _sph2cart(fr.Lon, fr.Lat, fr.Angle)

    n_samples = 10000
    ens = fr.build_ensemble(n_samples)
    rot_vecs = np.array([_sph2cart(item.Lon, item.Lat, item.Angle) for item in ens])
    sample_mean = np.mean(rot_vecs, axis=0)

    assert np.allclose(sample_mean, expected, atol=7e-3)


def test_finite_rotation_build_methods_reject_bad_covariance():
    bad_cov = pk.Covariance([-1.0, 0.0, 0.0, -1.0, 0.0, -1.0])
    fr_bad = pk.FiniteRotation(1.0, 2.0, 3.0, 4.0, bad_cov)

    with pytest.raises(ValueError):
        fr_bad.build_array(8)
    with pytest.raises(ValueError):
        fr_bad.build_ensemble(8)



#-- EulerVector build methods
def test_euler_vector_build_array_cartesian_and_spherical(ev):
    arr_cart = ev.build_array(20)
    arr_sph = ev.build_array(20, "spherical")
    assert isinstance(arr_cart, np.ndarray)
    assert isinstance(arr_sph, np.ndarray)
    assert arr_cart.shape == (3, 20)
    assert arr_sph.shape == (3, 20)


def test_euler_vector_build_ensemble(ev):
    ens = ev.build_ensemble(10)
    assert isinstance(ens, list)
    assert len(ens) == 10
    assert all(isinstance(item, pk.EulerVector) for item in ens)


def test_euler_vector_build_methods_fail_without_covariance():
    ev_no_cov = pk.EulerVector(1.0, 2.0, 3.0, (0.0, 0.0))
    with pytest.raises(TypeError):
        ev_no_cov.build_array(8)
    with pytest.raises(TypeError):
        ev_no_cov.build_ensemble(8)


def test_euler_vector_build_array_spherical_accuracy(ev):
    expected_mean = _sph2cart(ev.Lon, ev.Lat, ev.AngVelocity)

    n_samples = 10000
    samples_sph = ev.build_array(n_samples, "spherical")
    samples_sph_to_cart = _sph2cart(
        samples_sph[0], samples_sph[1], samples_sph[2]
    )

    sample_mean = np.mean(samples_sph_to_cart, axis=1)
    assert np.allclose(sample_mean, expected_mean, atol=7e-3)


def test_euler_vector_build_array_cartesian_accuracy(ev):
    expected_mean = _sph2cart(ev.Lon, ev.Lat, ev.AngVelocity)

    n_samples = 10000
    samples_cart = ev.build_array(n_samples, "cartesian")
    sample_mean = np.mean(samples_cart, axis=1)

    assert np.allclose(sample_mean, expected_mean, atol=7e-3)


def test_euler_vector_build_ensemble_accuracy(ev):
    expected_mean = _sph2cart(ev.Lon, ev.Lat, ev.AngVelocity)

    n_samples = 10000
    ens = ev.build_ensemble(n_samples)
    samples_cart = np.array([_sph2cart(item.Lon, item.Lat, item.AngVelocity) for item in ens]).T
    sample_mean = np.mean(samples_cart, axis=1)

    assert np.allclose(sample_mean, expected_mean, atol=7e-3)


def test_euler_vector_build_methods_reject_bad_covariance():
    bad_cov = pk.Covariance([-1.0, 0.0, 0.0, -1.0, 0.0, -1.0])
    ev_bad = pk.EulerVector(1.0, 2.0, 3.0, (0.0, 1.0), bad_cov)

    with pytest.raises(ValueError):
        ev_bad.build_array(8)
    with pytest.raises(ValueError):
        ev_bad.build_ensemble(8)
