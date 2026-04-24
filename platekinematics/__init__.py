"""Runtime loader setup for Windows extension dependencies."""

import os
from pathlib import Path

# Keep handles alive for the life of the process. If they are garbage-collected,
# Windows can remove the added DLL directories again.
_DLL_DIR_HANDLES = []


def _add_dll_dir_if_exists(path: Path) -> None:
    if not path.is_dir():
        return

    if hasattr(os, "add_dll_directory"):
        try:
            _DLL_DIR_HANDLES.append(os.add_dll_directory(str(path)))
            return
        except OSError:
            pass

    # Fallback for non-Windows or when add_dll_directory is unavailable.
    os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")


_pkg_dir = Path(__file__).resolve().parent
_repo_root = _pkg_dir.parent

_candidate_dirs = [
    _pkg_dir,
    _repo_root / "src" / "vcpkg" / "installed" / "x64-windows" / "bin",
]

_conda_prefix = os.environ.get("CONDA_PREFIX")
if _conda_prefix:
    _candidate_dirs.append(Path(_conda_prefix) / "Library" / "bin")

for _dir in _candidate_dirs:
    _add_dll_dir_if_exists(_dir)
