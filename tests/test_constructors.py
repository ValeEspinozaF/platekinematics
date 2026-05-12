import pytest
import numpy as np

import platekinematics as pk


COV_VALUES = [1.179e-8, -1.317e-9, -2.481e-9, 2.881e-9, -4.622e-9, 9.316e-9]


#-- Covariance constructor
def test_covariance_constructor(cov):
    cov1 = pk.Covariance(COV_VALUES) # construct from list
    assert cov1.C11 == pytest.approx(COV_VALUES[0])
    assert cov1.C12 == pytest.approx(COV_VALUES[1])
    assert cov1.C13 == pytest.approx(COV_VALUES[2])
    assert cov1.C22 == pytest.approx(COV_VALUES[3])
    assert cov1.C23 == pytest.approx(COV_VALUES[4])
    assert cov1.C33 == pytest.approx(COV_VALUES[5])

    cov2 = pk.Covariance(tuple(COV_VALUES)) # construct from tuple
    assert cov2.C11 == pytest.approx(COV_VALUES[0])

    cov3 = pk.Covariance() # default constructor
    assert cov3.C11 == pytest.approx(1.0)
    assert cov3.C12 == pytest.approx(0.0)


def test_covariance_bad_length_fails():
    with pytest.raises(ValueError):
        pk.Covariance([1.0, 2.0, 3.0])


def test_covariance_numpy_array_constructor():
    cov = pk.Covariance(np.array(COV_VALUES, dtype=float))
    assert cov.C11 == pytest.approx(COV_VALUES[0])
    assert cov.C12 == pytest.approx(COV_VALUES[1])
    assert cov.C13 == pytest.approx(COV_VALUES[2])
    assert cov.C22 == pytest.approx(COV_VALUES[3])
    assert cov.C23 == pytest.approx(COV_VALUES[4])
    assert cov.C33 == pytest.approx(COV_VALUES[5])


def test_covariance_stress_constructor_rebinding():
    slots = {}

    for i in range(5000):
        cov = pk.Covariance(COV_VALUES)
        renamed = cov
        old_name = cov
        cov = None

        assert renamed.C11 == pytest.approx(COV_VALUES[0])
        assert old_name.C33 == pytest.approx(COV_VALUES[5])

        slots[f"cov_{i % 16}"] = renamed

    for cov in slots.values():
        assert cov.C11 == pytest.approx(COV_VALUES[0])
        assert cov.C33 == pytest.approx(COV_VALUES[5])



#-- FiniteRotation constructor variants
def test_finite_rotation_constructor_variants(fr):
    fr1 = pk.FiniteRotation(1.0, 2.0, 3.0, 4.0) # no covariance
    assert fr1.Lon == pytest.approx(1.0)
    assert fr1.Lat == pytest.approx(2.0)
    assert fr1.Time == pytest.approx(4.0)
    assert fr1.Covariance is None

    fr2 = pk.FiniteRotation(1.0, 2.0, 3.0, 4.0, fr.Covariance) # construct with existing covariance
    assert fr2.Angle == pytest.approx(3.0)
    assert fr2.Covariance.C12 == pytest.approx(fr.Covariance.C12)

    fr3 = pk.FiniteRotation(1.0, 2.0, 3.0, 4.0, pk.Covariance(COV_VALUES)) # construct with new covariance
    assert fr3.Covariance.C33 == pytest.approx(COV_VALUES[-1])

def test_finite_rotation_bad_length_fails():
    with pytest.raises(TypeError):
        pk.FiniteRotation(1.0, 2.0, 3.0) # lon, lat, angle and time are required


def test_finite_rotation_list_repr_stable_with_numpy_inputs():
    fr_numpy = np.array([
        [1.0, 2.0, 3.0, 4.0, 1.179e-8, -1.317e-9, -2.481e-9, 2.881e-9, -4.622e-9, 9.316e-9],
        [2.0, 3.0, 4.0, 5.0, 1.179e-8, -1.317e-9, -2.481e-9, 2.881e-9, -4.622e-9, 9.316e-9],
    ], dtype=float)

    fr_list = []
    for row in fr_numpy:
        cov = pk.Covariance([row[4], row[5], row[6], row[7], row[8], row[9]])
        fr_list.append(pk.FiniteRotation(row[2], row[1], row[3], row[0], cov))

    rendered = repr(fr_list)
    assert "FiniteRot(" in rendered
    assert "Lon=" in rendered


def test_finite_rotation_stress_constructor_rebinding(cov):
    slots = {}

    for i in range(5000):
        fr = pk.FiniteRotation(10.0, 20.0, 3.0, 4.0, cov)
        renamed = fr
        old_name = fr
        fr = None

        assert renamed.Lon == pytest.approx(10.0)
        assert old_name.Angle == pytest.approx(3.0)

        slots[f"fr_{i % 16}"] = renamed

    for fr in slots.values():
        assert fr.Lat == pytest.approx(20.0)
        assert fr.Time == pytest.approx(4.0)



#-- EulerVector constructor variants
def test_euler_vector_constructor_variants(ev):
    ev1 = pk.EulerVector(1.0, 2.0, 3.0, (4.0, 5.0)) # no covariance
    assert ev1.Lon == pytest.approx(1.0)
    assert ev1.Lat == pytest.approx(2.0)
    assert ev1.TimeRange[0] == pytest.approx(4.0)
    assert ev1.Covariance is None

    ev2 = pk.EulerVector(1.0, 2.0, 3.0, (4.0, 5.0), ev.Covariance) # construct with existing covariance
    assert ev2.AngVelocity == pytest.approx(3.0)
    assert ev2.Covariance.C12 == pytest.approx(ev.Covariance.C12)

    ev3 = pk.EulerVector(1.0, 2.0, 3.0, (4.0, 5.0), pk.Covariance(COV_VALUES)) # construct with new covariance
    assert ev3.Covariance.C33 == pytest.approx(COV_VALUES[-1])


def test_euler_vector_bad_length_fails():
    with pytest.raises(TypeError):
        pk.EulerVector(1.0, 2.0, 3.0, 4.0) # time range must be a tuple of two elements
    with pytest.raises(TypeError):
        pk.EulerVector(1.0, 2.0, 3.0) # lon, lat, ang velocity and time range are required


def test_euler_vector_stress_constructor_rebinding(cov):
    slots = {}

    for i in range(5000):
        ev = pk.EulerVector(10.0, -20.0, 1.5, (0.0, 5.0), cov)
        renamed = ev
        old_name = ev
        ev = None

        assert renamed.Lon == pytest.approx(10.0)
        assert old_name.AngVelocity == pytest.approx(1.5)

        slots[f"ev_{i % 16}"] = renamed

    for ev in slots.values():
        assert ev.Lat == pytest.approx(-20.0)
        assert ev.TimeRange[1] == pytest.approx(5.0)



#-- Stat constructor variants
def test_stat_constructor_variants():
    s1 = pk.Stat(1.2, 0.4)
    assert s1.Mean == pytest.approx(1.2)
    assert s1.StDev == pytest.approx(0.4)

    s2 = pk.Stat([1.2, 0.4])    
    assert s2.Mean == pytest.approx(1.2)
    assert s2.StDev == pytest.approx(0.4)

    s3 = pk.Stat()
    assert s3.Mean == pytest.approx(0.0)
    assert s3.StDev == pytest.approx(0.0)


def test_stat_bad_length_fails():
    with pytest.raises(ValueError):
        pk.Stat([1.0, 0.5, 0.1]) # too many elements in list
    with pytest.raises(TypeError):
        pk.Stat(1.0, 0.5, 0.1) # too many arguments in constructor
    with pytest.raises(ValueError):
        pk.Stat([1.0]) # too few elements in list
    with pytest.raises(TypeError):
        pk.Stat(1.0) # too few arguments


def test_stat_stress_constructor_rebinding():
    slots = {}

    for i in range(5000):
        st = pk.Stat(2.5, 0.25)
        renamed = st
        old_name = st
        st = None

        assert renamed.Mean == pytest.approx(2.5)
        assert old_name.StDev == pytest.approx(0.25)

        slots[f"st_{i % 16}"] = renamed

    for st in slots.values():
        assert st.Mean == pytest.approx(2.5)
        assert st.StDev == pytest.approx(0.25)


#-- SurfaceVelocity constructor variants
def test_surface_velocity_constructor_variants():
    sv1 = pk.SurfaceVelocity(10.0, 20.0, 3.0)
    assert isinstance(sv1.TotalVel, float)
    assert sv1.Lon == pytest.approx(10.0)
    assert sv1.Lat == pytest.approx(20.0)
    assert sv1.TotalVel == pytest.approx(3.0)

    sv2 = pk.SurfaceVelocity(10.0, 20.0, 1.0, 2.0, 3.0, 45.0)
    assert isinstance(sv2.EastVel, float)
    assert sv2.Lon == pytest.approx(10.0)
    assert sv2.Lat == pytest.approx(20.0)
    assert sv2.EastVel == pytest.approx(1.0)
    assert sv2.NorthVel == pytest.approx(2.0)
    assert sv2.TotalVel == pytest.approx(3.0)
    assert sv2.Azimuth == pytest.approx(45.0)

    sv3 = pk.SurfaceVelocity(10.0, 20.0, [1.0, 0.1], [2.0, 0.2], [3.0, 0.3], [45.0, 1.0])
    assert isinstance(sv3.EastVel, pk.Stat)
    assert isinstance(sv3.NorthVel, pk.Stat)
    assert isinstance(sv3.TotalVel, pk.Stat)
    assert isinstance(sv3.Azimuth, pk.Stat)
    assert sv3.EastVel.Mean == pytest.approx(1.0)
    assert sv3.EastVel.StDev == pytest.approx(0.1)


def test_surface_velocity_bad_length_fails():
    with pytest.raises(TypeError):
        pk.SurfaceVelocity(10.0, 20.0) # too few arguments

    with pytest.raises(TypeError):
        pk.SurfaceVelocity(10.0, 20.0, 1.0, 2.0, 3.0, 4.0, 5.0) # too many arguments


def test_surface_velocity_stress_constructor_rebinding():
    slots = {}

    for i in range(5000):
        sv = pk.SurfaceVelocity(10.0, 20.0, 1.0, 2.0, 3.0, 45.0)
        renamed = sv
        old_name = sv
        sv = None

        assert renamed.Lon == pytest.approx(10.0)
        assert old_name.TotalVel == pytest.approx(3.0)

        slots[f"sv_{i % 16}"] = renamed

    for sv in slots.values():
        assert sv.Lat == pytest.approx(20.0)
        assert sv.Azimuth == pytest.approx(45.0)