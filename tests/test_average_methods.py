import pytest

from platekinematics import pk_structs as pk


def test_average_fr_from_list_and_matrix(fr):
    ens = fr.build_ensemble(24)
    out_list = pk.average_fr(ens)
    assert isinstance(out_list, pk.FiniteRotation)

    mats = fr.build_array(24)
    out_mat = pk.average_fr(mats)
    assert isinstance(out_mat, pk.FiniteRotation)


def test_average_ev_from_list_and_matrix(ev):
    ens = ev.build_ensemble(24)
    out_list = pk.average_ev(ens)
    assert isinstance(out_list, pk.EulerVector)

    coords = ev.build_array(24)
    out_arr = pk.average_ev(coords)
    assert isinstance(out_arr, pk.EulerVector)


def test_average_functions_reject_invalid_input():
    with pytest.raises(TypeError):
        pk.average_fr("not-an-ensemble")
    with pytest.raises(TypeError):
        pk.average_ev("not-an-ensemble")
