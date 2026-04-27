import pytest

from platekinematics import pk_structs as pk


COV_VALUES = [1.179e-8, -1.317e-9, -2.481e-9, 2.881e-9, -4.622e-9, 9.316e-9]
ZERO_COV_VALUES = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@pytest.fixture
def cov():
    return pk.Covariance(COV_VALUES)


@pytest.fixture
def fr(cov):
    return pk.FiniteRotation(139.9907, -61.4772, 0.2977, 0.773, cov)


@pytest.fixture
def ev(cov):
    return pk.EulerVector(3.0, 5.0, 2.0, (4.5, 3.5), cov)
