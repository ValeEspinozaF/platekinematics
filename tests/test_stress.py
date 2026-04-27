from platekinematics import pk_structs as pk


def test_stress_repeat_surface_velocity_calls_no_crash(cov):
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


def test_stress_repeat_average_calls_no_crash(fr, ev):
    fr_ens = fr.build_ensemble(10)
    ev_ens = ev.build_ensemble(10)

    for _ in range(300):
        fr_avg = pk.average_fr(fr_ens)
        ev_avg = pk.average_ev(ev_ens)
        assert isinstance(fr_avg, pk.FiniteRotation)
        assert isinstance(ev_avg, pk.EulerVector)
