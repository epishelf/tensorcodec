"""Build the py3-none-any fallback wheel: every Python module, no native extension.

Maturin always compiles the extension, so the package is staged with the same
[project] metadata and built by hatchling. Installers prefer the native wheels
where their tags match and fall back to this one elsewhere.
"""

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parents[1]
BUILD_SYSTEM = """[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/tensorcodec"]
"""


def project_metadata(text):
    """Return the [project] tables verbatim so the fallback cannot drift from the native wheels."""
    tables = re.split(r"(?m)^(?=\[)", text)
    project = "".join(table for table in tables if re.match(r"\[project[\].]", table))
    if tomllib.loads(project)["project"] != tomllib.loads(text)["project"]:
        raise RuntimeError("could not extract [project] metadata from pyproject.toml")
    return project


def stage(directory):
    shutil.copytree(
        ROOT / "src" / "tensorcodec",
        directory / "src" / "tensorcodec",
        ignore=shutil.ignore_patterns("_native*", "__pycache__", "*.pyc"),
    )
    shutil.copytree(ROOT / "licenses", directory / "licenses")
    for name in ("LICENSE", "README.md"):
        shutil.copy2(ROOT / name, directory / name)
    text = (ROOT / "pyproject.toml").read_text()
    (directory / "pyproject.toml").write_text(project_metadata(text) + "\n" + BUILD_SYSTEM)


def check(wheel):
    if not wheel.name.endswith("-py3-none-any.whl"):
        raise RuntimeError(f"unexpected wheel tag: {wheel.name}")
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
    if any(Path(name).name.startswith("_native") for name in names):
        raise RuntimeError("fallback wheel must not contain the native extension")
    if "tensorcodec/decoders/_images.py" not in names:
        raise RuntimeError("fallback wheel is missing the image decoders")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=ROOT / "dist")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        staging, built = Path(tmp) / "stage", Path(tmp) / "dist"
        stage(staging)
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", built, staging],
            check=True,
        )
        (wheel,) = built.glob("*.whl")
        check(wheel)
        target = args.out / wheel.name
        shutil.copy2(wheel, target)
    print(target)


if __name__ == "__main__":
    main()
