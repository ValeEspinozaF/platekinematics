from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from . import _core as pk

X_KEYS = ("x", "X")
Y_KEYS = ("y", "Y")
Z_KEYS = ("z", "Z")
LON_KEYS = ("lon", "long", "longitude", "Lon", "Long", "Longitude")
LAT_KEYS = ("lat", "latitude", "Lat", "Latitude")
ANGLE_KEYS = ("angle", "w", "Angle", "W")
VEL_KEYS = ("om", "omega", "Om", "Omega", "vel", "velocity", "angvel", "AngVelocity")
AGE_KEYS = ("age", "Age", "t", "T", "time", "Time")
AGE1_KEYS = ("age1", "Age1", "t1", "T1", "time1", "Time1")
AGE2_KEYS = ("age2", "Age2", "t2", "T2", "time2", "Time2")
C11_KEYS = ("a", "A", "c11", "C11", "cxx", "CXX", "Cxx", "cXX")
C12_KEYS = ("b", "B", "c12", "C12", "cxy", "CXY", "Cxy", "cXY")
C13_KEYS = ("c", "C", "c13", "C13", "cxz", "CXZ", "Cxz", "cXZ")
C22_KEYS = ("d", "D", "c22", "C22", "cyy", "CYY", "Cyy", "cYY")
C23_KEYS = ("e", "E", "c23", "C23", "cyz", "CYZ", "Cyz", "cYZ")
C33_KEYS = ("f", "F", "c33", "C33", "czz", "CZZ", "Czz", "cZZ")


class TXTParseError(ValueError):
    pass


def _split_fields(line: str, delimiter: str | None) -> list[str]:
    return line.strip().split(delimiter) if delimiter is not None else line.strip().split()


def _normalize_names(names: Sequence[str] | None) -> list[str] | None:
    if names is None:
        return None
    return [str(name).strip() for name in names]


def _first_match(names: Sequence[str], keys: Iterable[str]) -> int | None:
    for key in keys:
        try:
            return names.index(key)
        except ValueError:
            continue
    return None


def _read_numeric_table(
    file_path: str | Path,
    *,
    names: Sequence[str] | None,
    header: bool,
    skipstart: int,
    skipblanks: bool,
    comments: bool,
    comment_char: str,
    delimiter: str | None,
) -> tuple[list[list[float]], list[str]]:
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    if not header and names is None:
        raise TXTParseError("names or header must be provided")

    cleaned_lines: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for i, raw in enumerate(handle):
            if i < skipstart:
                continue

            line = raw.rstrip("\n")
            stripped = line.strip()

            if skipblanks and stripped == "":
                continue
            if comments and stripped.startswith(comment_char):
                continue
            if stripped == "":
                continue

            cleaned_lines.append(line)

    if not cleaned_lines:
        raise TXTParseError("No readable data lines found")

    parsed_names = _normalize_names(names)
    data_lines = cleaned_lines

    if header:
        header_tokens = _split_fields(cleaned_lines[0], delimiter)
        if parsed_names is None:
            parsed_names = [str(token).strip() for token in header_tokens]
        data_lines = cleaned_lines[1:]

    if parsed_names is None:
        raise TXTParseError("names or header must be provided")

    rows: list[list[float]] = []
    for row_idx, line in enumerate(data_lines, start=1):
        tokens = _split_fields(line, delimiter)
        if not tokens:
            continue

        if len(tokens) != len(parsed_names):
            raise TXTParseError(
                f"Row {row_idx} has {len(tokens)} columns but {len(parsed_names)} names were provided"
            )

        try:
            rows.append([float(token) for token in tokens])
        except ValueError as exc:
            raise TXTParseError(f"Row {row_idx} contains non-numeric values") from exc

    if not rows:
        raise TXTParseError("No numeric data rows found")

    return rows, parsed_names


def _build_covariance(row: Sequence[float], names: Sequence[str]) -> pk.Covariance | None:
    i11 = _first_match(names, C11_KEYS)
    i12 = _first_match(names, C12_KEYS)
    i13 = _first_match(names, C13_KEYS)
    i22 = _first_match(names, C22_KEYS)
    i23 = _first_match(names, C23_KEYS)
    i33 = _first_match(names, C33_KEYS)

    if None in (i11, i12, i13, i22, i23, i33):
        return None

    return pk.Covariance([row[i11], row[i12], row[i13], row[i22], row[i23], row[i33]])


def _parse_as_finite_rotation(rows: list[list[float]], names: list[str]) -> list[pk.FiniteRotation]:
    i_lon = _first_match(names, LON_KEYS)
    i_lat = _first_match(names, LAT_KEYS)
    i_ang = _first_match(names, ANGLE_KEYS)
    i_age = _first_match(names, AGE_KEYS)

    if None in (i_lon, i_lat, i_ang):
        raise TXTParseError("Missing required columns for FiniteRotation: lon/lat/angle")

    out: list[pk.FiniteRotation] = []
    for row in rows:
        cov = _build_covariance(row, names)
        time_val = row[i_age] if i_age is not None else 0.0
        if cov is None:
            out.append(pk.FiniteRotation(row[i_lon], row[i_lat], row[i_ang], time_val))
        else:
            out.append(pk.FiniteRotation(row[i_lon], row[i_lat], row[i_ang], time_val, cov))

    return out


def _parse_as_euler_vector(rows: list[list[float]], names: list[str]) -> list[pk.EulerVector]:
    i_lon = _first_match(names, LON_KEYS)
    i_lat = _first_match(names, LAT_KEYS)
    i_vel = _first_match(names, VEL_KEYS)
    i_age1 = _first_match(names, AGE1_KEYS)
    i_age2 = _first_match(names, AGE2_KEYS)

    if None in (i_lon, i_lat, i_vel):
        raise TXTParseError("Missing required columns for EulerVector: lon/lat/velocity")

    out: list[pk.EulerVector] = []
    for row in rows:
        cov = _build_covariance(row, names)
        time_range = (
            (row[i_age1], row[i_age2])
            if i_age1 is not None and i_age2 is not None
            else (0.0, 0.0)
        )

        if cov is None:
            out.append(pk.EulerVector(row[i_lon], row[i_lat], row[i_vel], time_range))
        else:
            out.append(pk.EulerVector(row[i_lon], row[i_lat], row[i_vel], time_range, cov))

    return out


def read_txt(
    file_path: str | Path,
    struct_type: type,
    *,
    names: Sequence[str] | None = None,
    header: bool = False,
    skipstart: int = 0,
    skipblanks: bool = True,
    comments: bool = True,
    comment_char: str = "#",
    delimiter: str | None = None,
):
    """Load a TXT table into a list of platekinematics objects.

    Supported struct types are platekinematics.FiniteRotation and platekinematics.EulerVector.
    Column names can be provided with `names` or taken from a header row.
    """

    rows, parsed_names = _read_numeric_table(
        file_path,
        names=names,
        header=header,
        skipstart=skipstart,
        skipblanks=skipblanks,
        comments=comments,
        comment_char=comment_char,
        delimiter=delimiter,
    )

    if struct_type is pk.FiniteRotation:
        return _parse_as_finite_rotation(rows, parsed_names)
    if struct_type is pk.EulerVector:
        return _parse_as_euler_vector(rows, parsed_names)

    raise TXTParseError("Invalid struct_type. Use platekinematics.FiniteRotation or platekinematics.EulerVector")


def read_txt_as_dict(
    file_path: str | Path,
    *,
    names: Sequence[str] | None = None,
    header: bool = False,
    skipstart: int = 0,
    skipblanks: bool = True,
    comments: bool = True,
    comment_char: str = "#",
    delimiter: str | None = None,
) -> dict[str, list[float]]:
    """Load a numeric TXT table into a dictionary keyed by column name."""

    rows, parsed_names = _read_numeric_table(
        file_path,
        names=names,
        header=header,
        skipstart=skipstart,
        skipblanks=skipblanks,
        comments=comments,
        comment_char=comment_char,
        delimiter=delimiter,
    )

    out = {name: [] for name in parsed_names}
    for row in rows:
        for i, name in enumerate(parsed_names):
            out[name].append(row[i])
    return out
