from pathlib import Path

import platekinematics as pkg
from platekinematics import pk_structs as pk


def _write_tmp(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def test_read_txt_finiterotation_header(tmp_path):
    file_path = _write_tmp(
        tmp_path / "fr.txt",
        "lon lat angle time c11 c12 c13 c22 c23 c33\n"
        "10 20 5 7 1 0 0 1 0 1\n"
        "30 40 6 8 1 0 0 1 0 1\n",
    )

    out = pkg.read_txt(file_path, pk.FiniteRotation, header=True)

    assert len(out) == 2
    assert isinstance(out[0], pk.FiniteRotation)
    assert out[0].Lon == 10.0
    assert out[0].Time == 7.0
    assert out[0].Covariance.C11 == 1.0


def test_read_txt_eulervector_names_no_header(tmp_path):
    file_path = _write_tmp(
        tmp_path / "ev.txt",
        "10 20 0.3 12 5\n"
        "30 40 0.4 8 0\n",
    )

    out = pkg.read_txt(
        file_path,
        pk.EulerVector,
        names=["lon", "lat", "omega", "time1", "time2"],
        header=False,
    )

    assert len(out) == 2
    assert isinstance(out[0], pk.EulerVector)
    assert out[0].TimeRange == (12.0, 5.0)
    assert out[1].AngVelocity == 0.4


def test_read_txt_is_available_on_pk_structs(tmp_path):
    file_path = _write_tmp(
        tmp_path / "fr_simple.txt",
        "lon lat angle time\n"
        "10 20 5 7\n",
    )

    out = pk.read_txt(file_path, pk.FiniteRotation, header=True)

    assert len(out) == 1
    assert isinstance(out[0], pk.FiniteRotation)
