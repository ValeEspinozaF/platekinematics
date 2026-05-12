import math

import platekinematics as pk


def _fr(lon, lat, angle, time):
    return pk.FiniteRotation(float(lon), float(lat), float(angle), float(time))


def test_to_euler_vector_single_finiterotation():
    fr = _fr(30, -10, 12, 5)
    ev = pk.to_euler_vector(fr)

    assert isinstance(ev, pk.EulerVector)
    assert ev.TimeRange == (5.0, 0.0)
    assert math.isfinite(ev.Lon)
    assert math.isfinite(ev.Lat)
    assert math.isfinite(ev.AngVelocity)


def test_to_euler_vector_single_reverse():
    fr = _fr(30, -10, 12, 5)
    ev = pk.to_euler_vector(fr, reverse_rot=True)

    assert isinstance(ev, pk.EulerVector)
    assert ev.TimeRange == (0.0, 5.0)
    assert math.isfinite(ev.AngVelocity)


def test_to_euler_vector_pair_finiterotation():
    fr1 = _fr(20, 10, 8, 10)
    fr2 = _fr(25, 12, 9, 5)

    ev = pk.to_euler_vector(fr1, fr2)
    assert isinstance(ev, pk.EulerVector)
    assert ev.TimeRange == (10.0, 5.0)


def test_to_euler_vector_samples_list():
    frs = [_fr(20.0, 10.0, 8.0, 6.0), _fr(20.2, 10.1, 8.1, 6.0), _fr(19.9, 10.3, 7.9, 6.0)]

    ev = pk.to_euler_vector(frs)
    assert isinstance(ev, pk.EulerVector)
    assert ev.TimeRange == (6.0, 0.0)


def test_to_euler_vector_list_stage_series():
    frs = [_fr(20, 10, 8, 10), _fr(25, 12, 9, 5), _fr(30, 14, 10, 0)]

    out = pk.to_euler_vector_list(frs)
    assert isinstance(out, list)
    assert len(out) == len(frs)
    assert all(isinstance(ev, pk.EulerVector) for ev in out)
