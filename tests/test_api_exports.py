from platekinematics import pk_structs as pk


def test_api_export_names_stable():
    expected = {
        "Covariance",
        "Stat",
        "FiniteRotation",
        "EulerVector",
        "SurfaceVelocity",
        "average_fr",
        "average_ev",
        "calculate_surface_velocity",
        "calculate_mean_surface_velocity",
        "to_euler_vector",
        "to_euler_vector_list",
    }
    for name in expected:
        assert hasattr(pk, name), f"Missing API name: {name}"