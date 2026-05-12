import pytest
import numpy as np

import platekinematics as pk


#-- FiniteRotation average methods
def test_average_fr_from_array(fr):
    mats = fr.build_array(24)
    out_fr = pk.average_fr(mats)
    assert isinstance(out_fr, pk.FiniteRotation)


def test_average_fr_from_array_accuracy(fr):
    expected = (fr.Lon, fr.Lat, fr.Angle)
    mats = fr.build_array(10000)
    out_fr = pk.average_fr(mats)
    out_tuple = (out_fr.Lon, out_fr.Lat, out_fr.Angle)

    assert np.allclose(out_tuple, expected, atol=7e-2)


def test_average_fr_from_list(fr):
    ens = fr.build_ensemble(24)
    out_list = pk.average_fr(ens)
    assert isinstance(out_list, pk.FiniteRotation)


def test_average_fr_from_ensemble_accuracy(fr):
    expected = (fr.Lon, fr.Lat, fr.Angle)
    ens = fr.build_ensemble(10000)
    out_list = pk.average_fr(ens)
    out_tuple = (out_list.Lon, out_list.Lat, out_list.Angle)

    assert np.allclose(out_tuple, expected, atol=7e-2)


@pytest.mark.parametrize(
    "bad_input",
    [
        "not-an-ensemble",
        None,
        42,
        3.14,
        {"a": 1},
        object(),
        np.array([1.0, 2.0]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    ],
)
def test_average_fr_reject_invalid_input(bad_input):
    with pytest.raises(TypeError):
        pk.average_fr(bad_input)



#-- EulerVector average methods
def test_average_ev_from_array(ev):
    coords = ev.build_array(24)
    out_arr = pk.average_ev(coords)
    assert isinstance(out_arr, pk.EulerVector)


def test_average_ev_from_array_accuracy(ev):
    expected = (ev.Lon, ev.Lat, ev.AngVelocity)
    coords = ev.build_array(10000)
    out_arr = pk.average_ev(coords)
    out_tuple = (out_arr.Lon, out_arr.Lat, out_arr.AngVelocity)

    assert np.allclose(out_tuple, expected, atol=7e-2)


def test_average_ev_from_ensemble(ev):
    ens = ev.build_ensemble(24)
    out_list = pk.average_ev(ens)
    assert isinstance(out_list, pk.EulerVector)


def test_average_ev_from_ensemble_accuracy(ev):
    expected = (ev.Lon, ev.Lat, ev.AngVelocity)
    ens = ev.build_ensemble(10000)
    out_list = pk.average_ev(ens)
    out_tuple = (out_list.Lon, out_list.Lat, out_list.AngVelocity)

    assert np.allclose(out_tuple, expected, atol=7e-2)


@pytest.mark.parametrize(
    "bad_input",
    [
        "not-an-ensemble",
        None,
        42,
        3.14,
        {"a": 1},
        object(),
        np.array([1.0, 2.0]),
        np.array([[1.0, 2.0], [3.0, 4.0]]),
    ],
)
def test_average_ev_reject_invalid_input(bad_input):
    with pytest.raises(TypeError):
        pk.average_ev(bad_input)


def test_stress_repeat_average_calls_no_crash(fr, ev):
    fr_ens = fr.build_ensemble(10)
    ev_ens = ev.build_ensemble(10)

    for _ in range(300):
        fr_avg = pk.average_fr(fr_ens)
        ev_avg = pk.average_ev(ev_ens)
        assert isinstance(fr_avg, pk.FiniteRotation)
        assert isinstance(ev_avg, pk.EulerVector)