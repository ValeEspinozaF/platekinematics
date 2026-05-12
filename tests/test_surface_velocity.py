import pytest
import numpy as np

import platekinematics as pk


#-- Single Euler vector tests
# calculate_surface_velocity(ev, lon, lat[, n_size]) -> SurfaceVelocity
# calculate_surface_velocity(ev, lons, lats[, n_size]) -> list
class TestSingleEulerVector:
    def test_output_type(self, ev):
        out1 = pk.calculate_surface_velocity(ev, 10.0, 45.0, 100)

        assert isinstance(out1, pk.SurfaceVelocity)
        assert isinstance(out1.EastVel, pk.Stat)
        assert isinstance(out1.NorthVel, pk.Stat)
        assert isinstance(out1.TotalVel, pk.Stat)
        assert isinstance(out1.Azimuth, pk.Stat)

        out2 = pk.calculate_surface_velocity(ev, [10.0, 10.0], [45.0, 45.0], 100)

        assert isinstance(out2, list)
        assert len(out2) == 2
        for out in out2:
            assert isinstance(out, pk.SurfaceVelocity)
            assert isinstance(out.EastVel, pk.Stat)
            assert isinstance(out.NorthVel, pk.Stat)
            assert isinstance(out.TotalVel, pk.Stat)
            assert isinstance(out.Azimuth, pk.Stat)

    def test_no_covariance_output_type(self, ev_zero_cov, ev_no_cov):
        out1 = pk.calculate_surface_velocity(ev_no_cov, 10.0, 45.0)
        out2 = pk.calculate_surface_velocity(ev_zero_cov, 10.0, 45.0)

        for out in [out1, out2]:
            assert isinstance(out, pk.SurfaceVelocity)
            assert isinstance(out.EastVel, float)
            assert isinstance(out.NorthVel, float)
            assert isinstance(out.TotalVel, float)
            assert isinstance(out.Azimuth, float)

        out3 = pk.calculate_surface_velocity(ev_no_cov, [10.0, 10.0], [45.0, 45.0])
        out4 = pk.calculate_surface_velocity(ev_zero_cov, [10.0, 10.0], [45.0, 45.0])

        for outs in [out3, out4]:
            assert isinstance(outs, list)
            assert len(outs) == 2
            for out in outs:
                assert isinstance(out, pk.SurfaceVelocity)
                assert isinstance(out.EastVel, float)
                assert isinstance(out.NorthVel, float)
                assert isinstance(out.TotalVel, float)
                assert isinstance(out.Azimuth, float)

    def test_rejects_bad_inputs(self, ev):
        with pytest.raises(TypeError):
            pk.calculate_surface_velocity(123, 10.0, 20.0)

        with pytest.raises(ValueError):
            pk.calculate_surface_velocity(ev, [10.0, 20.0], [10.0], 100)

    def test_rejects_no_covariance_bad_inputs(self, ev_no_cov, ev_zero_cov):
        with pytest.raises(ValueError):
            pk.calculate_surface_velocity(ev_no_cov, 10.0, 45.0, 100)

        with pytest.raises(ValueError):
            pk.calculate_surface_velocity(ev_zero_cov, 10.0, 45.0, 100)

    def test_accuracy(self, ev):
        ev1 = pk.EulerVector(45.0, 20.0, 0.5, (0.0, 0.0))
        out1 = pk.calculate_surface_velocity(ev1, [10.0], [45.0])
        assert out1[0].EastVel == pytest.approx(-1.666, abs=1e-2)
        assert out1[0].NorthVel == pytest.approx(-2.995, abs=1e-2)
        assert out1[0].TotalVel == pytest.approx(3.427, abs=1e-2)
        assert out1[0].Azimuth == pytest.approx(-150.9, abs=1e-1)

        out2 = pk.calculate_surface_velocity(ev, [10.0], [45.0], 10000)
        assert out2[0].EastVel.Mean == pytest.approx(-14.113, abs=1e-2)
        assert out2[0].NorthVel.Mean == pytest.approx(2.698, abs=1e-2)
        assert out2[0].TotalVel.Mean == pytest.approx(14.369, abs=1e-2)
        assert out2[0].Azimuth.Mean == pytest.approx(-79.176, abs=1e-1)

        out3 = ev.calculate_surface_velocity(10.0, 45.0, 10000)
        assert out3.EastVel.Mean == out2[0].EastVel.Mean
        assert out3.NorthVel.Mean == out2[0].NorthVel.Mean
        assert out3.TotalVel.Mean == out2[0].TotalVel.Mean
        assert out3.Azimuth.Mean == out2[0].Azimuth.Mean

    def test_no_covariance_accuracy(self, ev_no_cov, ev_zero_cov):
        out1 = pk.calculate_surface_velocity(ev_no_cov, 10.0, 45.0)
        out2 = pk.calculate_surface_velocity(ev_no_cov, [10.0], [45.0])
        out3 = pk.calculate_surface_velocity(ev_zero_cov, 10.0, 45.0)
        out4 = pk.calculate_surface_velocity(ev_zero_cov, [10.0], [45.0])

        for out in [out1, out2[0], out3, out4[0]]:
            assert isinstance(out, pk.SurfaceVelocity)
            assert out.EastVel == pytest.approx(-14.113, abs=1e-2)
            assert out.NorthVel == pytest.approx(2.698, abs=1e-2)
            assert out.TotalVel == pytest.approx(14.369, abs=1e-2)
            assert out.Azimuth == pytest.approx(-79.176, abs=1e-1)


#-- Euler vector method tests
# ev.calculate_surface_velocity(lon, lat[, n_size]) -> SurfaceVelocity
# ev.calculate_surface_velocity(lons, lats[, n_size]) -> list
class TestEulerVectorMethod:
    def test_single_output_type(self, ev, ev_no_cov, ev_zero_cov):
        out1 = ev.calculate_surface_velocity(10.0, 45.0, 10) 
        assert isinstance(out1, pk.SurfaceVelocity)
        assert isinstance(out1.EastVel, pk.Stat)

        out2 = ev_no_cov.calculate_surface_velocity(10.0, 45.0) 
        out3 = ev_zero_cov.calculate_surface_velocity(10.0, 45.0)
        for out in [out2, out3]:
            assert isinstance(out, pk.SurfaceVelocity)
            assert isinstance(out.EastVel, float)
        
    def test_sequence_output_type(self, ev, ev_no_cov, ev_zero_cov):
        out1 = ev.calculate_surface_velocity([10.0, 20.0], [45.0, 30.0], 10)
        assert isinstance(out1, list)
        assert len(out1) == 2
        assert all(isinstance(item, pk.SurfaceVelocity) for item in out1)
        assert all(isinstance(item.EastVel, pk.Stat) for item in out1)

        out2 = ev_no_cov.calculate_surface_velocity([10.0, 20.0], [45.0, 30.0])
        out3 = ev_zero_cov.calculate_surface_velocity([10.0, 20.0], [45.0, 30.0])
        for out in [out2, out3]:
            assert isinstance(out, list)
            assert len(out) == 2
            assert all(isinstance(item, pk.SurfaceVelocity) for item in out)
            assert all(isinstance(item.EastVel, float) for item in out) 

    def test_rejects_bad_inputs(self, ev):
        with pytest.raises(ValueError):
            ev.calculate_surface_velocity([10.0, 20.0], [45.0], 10)

    def test_rejects_no_covariance_bad_inputs(self, ev_no_cov, ev_zero_cov):
        with pytest.raises(ValueError):
            ev_no_cov.calculate_surface_velocity(10.0, 45.0, 10)

        with pytest.raises(ValueError):
            ev_zero_cov.calculate_surface_velocity(10.0, 45.0, 10)


#-- Ensemble of Euler vectors tests
# calculate_surface_velocity(ev_ensemble, lon, lat[, n_size]) -> tuple
# calculate_surface_velocity(ev_ensemble, lons, lats[, n_size]) -> list
class TestEnsembleEulerVectors:
    def test_shape(self, ev):
        n_size = 100
        ev_ens = ev.build_ensemble(n_size)
        out = pk.calculate_surface_velocity(ev_ens, 10.0, 45.0)

        assert isinstance(out, tuple)
        assert len(out) == 4

        for arr in out:
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (n_size,)

    def test_with_nsize_shape(self, ev):
        ev_ens = ev.build_ensemble(100)

        out = pk.calculate_surface_velocity(ev_ens, 10.0, 45.0, 10)
        assert isinstance(out, tuple)
        assert len(out) == 4

        for arr in out:
            assert isinstance(arr, np.ndarray)
            assert arr.shape == (10,)
            assert isinstance(arr[0], float)

    def test_with_nsize_bad_inputs(self, ev):
        ev_ens = ev.build_ensemble(8)

        with pytest.raises(ValueError):
            pk.calculate_surface_velocity(ev_ens, 10.0, 45.0, 16)

        with pytest.raises(TypeError):
            pk.calculate_surface_velocity(ev_ens, [10.0], [45.0], 4)


class TestMeanSurfaceVelocity:
    def test_calculate_mean_surface_velocity(self, ev):
        ev_ens = ev.build_ensemble(64)
        sv = pk.calculate_mean_surface_velocity(ev_ens, 10.0, 45.0)

        assert isinstance(sv, pk.SurfaceVelocity)
        assert isinstance(sv.EastVel, pk.Stat)
        assert isinstance(sv.NorthVel, pk.Stat)
        assert isinstance(sv.TotalVel, pk.Stat)
        assert isinstance(sv.Azimuth, pk.Stat)


class TestStress:
    def test_repeat_surface_velocity_calls_no_crash(self, cov):
        ev_cov = pk.EulerVector(45.0, 20.0, 0.5, (0.0, 0.0), cov)
        ev_no_cov = pk.EulerVector(45.0, 20.0, 0.5, (0.0, 0.0))
        ens = ev_cov.build_ensemble(8)

        for _ in range(200):
            out_a = pk.calculate_surface_velocity(ens, 10.0, 20.0)
            assert len(out_a) == 4

            out_b = pk.calculate_surface_velocity(ev_no_cov, [10.0, 20.0], [15.0, 25.0])
            assert len(out_b) == 2

            out_c = pk.calculate_surface_velocity(ev_cov, [10.0], [15.0], 64)
            assert len(out_c) == 1