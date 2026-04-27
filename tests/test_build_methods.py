import numpy as np
import pytest

from platekinematics import pk_structs as pk


def test_finite_rotation_build_array_and_ensemble(fr):
    mats = fr.build_array(16)
    assert isinstance(mats, np.ndarray)
    assert mats.shape == (16, 3, 3)

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
