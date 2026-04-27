import math

import pytest

from platekinematics import spherical_functions as sf


def test_spherical_functions_basic_roundtrip():
    rad = sf.to_radians(180.0)
    deg = sf.to_degrees(math.pi)
    assert rad == pytest.approx(math.pi)
    assert deg == pytest.approx(180.0)

    lon, lat, mag = sf.cart2sph(*sf.sph2cart(10.0, -30.0, 2.0))
    assert lon == pytest.approx(10.0, rel=1e-12)
    assert lat == pytest.approx(-30.0, rel=1e-12)
    assert mag == pytest.approx(2.0, rel=1e-12)