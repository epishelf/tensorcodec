"""tensorcodec and tensorcodec-native are released in lockstep from one commit."""

import re
from pathlib import Path

import pytest
from packaging.markers import default_environment
from packaging.requirements import Requirement

tomllib = pytest.importorskip("tomllib")
ROOT = Path(__file__).resolve().parents[1]


def native_requirements(project):
    groups = [project["dependencies"], *project["optional-dependencies"].values()]
    requirements = [Requirement(item) for group in groups for item in group]
    return [r for r in requirements if r.name == "tensorcodec-native"]


def test_versions_are_locked():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    version = project["version"]
    cargo = tomllib.loads((ROOT / "native/Cargo.toml").read_text())["package"]["version"]
    init = re.search(r'^__version__ = "(.+)"$', (ROOT / "src/tensorcodec/__init__.py").read_text(), re.MULTILINE)[1]
    assert cargo == init == version
    pins = native_requirements(project)
    assert len(pins) == 2
    assert all(str(r.specifier) == f"=={version}" for r in pins)


def test_native_license_matches_project_license():
    assert (ROOT / "native/LICENSE").read_bytes() == (ROOT / "LICENSE").read_bytes()


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        ({"sys_platform": "linux", "platform_machine": "x86_64"}, True),
        ({"sys_platform": "linux", "platform_machine": "aarch64"}, True),
        ({"sys_platform": "linux", "platform_machine": "armv7l"}, False),
        ({"sys_platform": "linux", "platform_machine": "x86_64", "platform_python_implementation": "PyPy"}, False),
        ({"sys_platform": "darwin", "platform_machine": "arm64", "platform_release": "23.0.0"}, True),
        ({"sys_platform": "darwin", "platform_machine": "arm64", "platform_release": "25.1.0"}, True),
        ({"sys_platform": "darwin", "platform_machine": "arm64", "platform_release": "22.6.0"}, False),
        ({"sys_platform": "darwin", "platform_machine": "x86_64", "platform_release": "24.0.0"}, False),
        ({"sys_platform": "win32", "platform_machine": "AMD64", "platform_release": "2025Server"}, False),
    ],
)
def test_native_marker_matches_published_wheels(environment, expected):
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    (default,) = [r for r in native_requirements(project) if r.marker is not None]
    base = default_environment() | {"platform_python_implementation": "CPython", "platform_release": "6.8.0-azure"}
    assert default.marker.evaluate(base | environment) is expected
