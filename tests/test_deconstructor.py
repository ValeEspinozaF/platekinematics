import numpy as np
import pytest

from platekinematics import pk_structs as pk


COV_VALUES = [1.179e-8, -1.317e-9, -2.481e-9, 2.881e-9, -4.622e-9, 9.316e-9]


def test_covariance_to_numpy_roundtrip(cov):
    arr_cov = cov.to_numpy()
    assert isinstance(arr_cov, np.ndarray)
    assert arr_cov.shape == (6,)
    np.testing.assert_allclose(arr_cov, np.array(COV_VALUES), rtol=0, atol=1e-16)


#-- FiniteRotation to_numpy method
def test_finite_rotation_to_numpy_shapes(fr):
    arr_fr = fr.to_numpy()
    assert isinstance(arr_fr, np.ndarray)
    assert arr_fr.shape == (10,)

    fr_no_cov = pk.FiniteRotation(1.0, 2.0, 3.0, 4.0)
    arr_fr_nocov = fr_no_cov.to_numpy()
    assert arr_fr_nocov.shape == (4,)


#-- EulerVector to_numpy method
def test_euler_vector_to_numpy_shapes(ev):
    arr_ev = ev.to_numpy()
    assert isinstance(arr_ev, np.ndarray)
    assert arr_ev.shape == (11,)

    ev_no_cov = pk.EulerVector(1.0, 2.0, 3.0, (0.0, 0.0))
    arr_ev_nocov = ev_no_cov.to_numpy()
    assert arr_ev_nocov.shape == (5,)