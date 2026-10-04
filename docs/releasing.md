# Publishing TensorCodec

Release version: `0.3.0`. One release publishes two PyPI projects from the same
commit and version:

| Project | Contents | Build | Distributions |
| --- | --- | --- | --- |
| `tensorcodec` | Python API, image codecs (import `tensorcodec`) | hatchling (`uv build`) | `py3-none-any` wheel, sdist |
| `tensorcodec-av` | Rust/FFmpeg audio/video decoder extension (import `tensorcodec_av`) | maturin (`av/`) | platform wheels, sdist |

`tensorcodec` requires `tensorcodec-av==<its own version>` behind an environment
marker matching the `tensorcodec-av` wheel tags. `tests/test_versions.py` fails CI unless
`pyproject.toml`, `av/Cargo.toml` (the `tensorcodec-av` version source), `av/Cargo.lock`, the pin and
`tensorcodec.__version__` agree. At runtime `VideoDecoder`/`AudioDecoder` reject a
`tensorcodec-av` whose version differs.

`tensorcodec-av` wheels target Linux x86_64 and ARM64 (aarch64), glibc 2.17+, CPython 3.10+
(abi3). NumPy must also provide a compatible wheel for the selected Python/glibc
pair. The wheel bundles shared FFmpeg 7.1.5 and OpenSSL 3.5.9 LTS; its only Python
runtime dependency is NumPy. macOS 14+ ARM64 wheels bundle the same minimal
runtime. Windows wheels are not provided; there, `tensorcodec` provides image
codecs only. Bundled-library notices ship only with `tensorcodec-av`
(`av/licenses/`).

To release, set the new version in `pyproject.toml` (project version and the
`tensorcodec-av` pin), `av/Cargo.toml` (then `cargo update -p
tensorcodec-av --manifest-path av/Cargo.toml` to refresh the lock file)
and `src/tensorcodec/__init__.py`.

## Trusted publisher configuration

The PyPI project is already registered. Its GitHub Trusted Publisher uses:

| Field | Value |
| --- | --- |
| PyPI project names | `tensorcodec`, `tensorcodec-av` |
| GitHub owner | `epishelf` |
| Repository | `tensorcodec` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

Both projects need this publisher. `tensorcodec-av` does not exist on PyPI yet:
before its first release, add it as a pending publisher (PyPI account → Publishing)
with the same fields. Manage this configuration in each project's PyPI Publishing settings when moving
or renaming the repository or workflow. Publishing uses GitHub OIDC; no API token
is needed. Repository visibility does not need to change for a release.

## Workflows

Each distribution has one reusable workflow; CI and the release call both.

| Workflow | Jobs | Artifacts |
| --- | --- | --- |
| `build-python.yml` | `build`: `tensorcodec` wheel and sdist (`python -m build`, `twine check --strict`); `test-without-av`: installs it on Windows and Intel macOS without `tensorcodec-av` and runs the image tests (OpenCV 5 and the 4.12 lower bound) | `dist-python` |
| `build-av.yml` | `linux` (x86_64, aarch64: manylinux2014 via maturin-action), `macos` (arm64), `sdist`; each wheel job validates with the `dist-python` wheel, so callers run `build-python.yml` first | `dist-av-linux-{x86_64,aarch64}`, `dist-av-macos-arm64`, `dist-av-sdist`, `wheel-size-linux-*` |
| `ci.yml` | `python`, `av`, plus `test` (development build against conda-forge FFmpeg, Clippy, oracle comparison) | — |
| `publish.yml` | `python`, `av`, `check` (one version, exactly the six expected files), `publish`, `update-size-docs` | — |

`ci.yml` runs `av` on every push, including `main`, and on pull requests that
touch `av/`, `scripts/`, `tests/`, workflows or `pyproject.toml`; a release
commit has therefore already passed the same `tensorcodec-av` builds and validation.

## Release

Run the **Publish to PyPI** workflow on `main`. It runs both build workflows, checks the
release set and uploads through PyPI Trusted Publishing using the existing GitHub
`pypi` environment. `tensorcodec-av` is uploaded first, so the exact pin in
`tensorcodec` never points at a missing release; if the second upload fails, upload
the `dist-python` artifact with `twine upload` rather than rerunning the whole
workflow. Publication fails if authorization is missing, tests fail, or the version
has already been uploaded.

```sh
gh workflow run publish.yml --repo epishelf/tensorcodec --ref main
```

For a build and full validation without uploading, pass `--field publish=false`;
the run's `dist-*` artifacts are then the release candidates.

Check the workflow, https://pypi.org/project/tensorcodec/ and
https://pypi.org/project/tensorcodec-av/ before reporting success. Verify a fresh
`uv pip install tensorcodec==<version>` pulls the matching `tensorcodec-av` and
decodes video without Torch/PyAV on both Linux architectures and macOS arm64, and
that it installs without `tensorcodec-av` on Windows. Update the version before
subsequent releases; PyPI versions cannot be overwritten.

The local Linux build is reproducible using `scripts/build_linux_wheel.sh` inside
`quay.io/pypa/manylinux2014_x86_64` or
`quay.io/pypa/manylinux2014_aarch64` with Rust, maturin, libclang, NASM and Perl.
Both native source archives are version- and checksum-pinned. Their licensing
and source links are recorded in `av/licenses/README.md`.

## Development versus release builds

- The `test` job uses prebuilt conda-forge FFmpeg 7.1.1 through Pixi, including its
  headers and shared libraries. It builds only the `tensorcodec-av` extension
  and installs `tensorcodec` from the checkout.
- Release wheels (`build-av.yml`) use the smaller LGPL FFmpeg 7.1.5 build plus
  OpenSSL 3.5.9. Their native prefix is cached by architecture, glibc baseline and
  build-script checksums. This preserves the wheel's codec set, dependency size and
  licensing rather than bundling the full conda-forge dependency graph.
- Each repaired Linux wheel is installed with the `tensorcodec` wheel on glibc 2.17
  with Python 3.10 and 3.13 and decodes video/audio without Torch, PyAV or a system
  FFmpeg. Python 3.10 also checks the minimum NumPy line (1.26.4). The x86_64 and
  ARM64 runners also run the full pinned playback oracle comparison; the macOS job
  runs the test suite and clean Python 3.10/3.13 environments.
- The fixture CLI can be FFmpeg 6 or 7; fixtures explicitly remove auxiliary
  sentinel packets.

## Size checks and published comparison

Final repaired `tensorcodec-av` wheels must stay within 15 MiB download and 35 MiB unpacked per
architecture. The `linux` job checks `packaging/size-policy.json` and uploads a
separate `wheel-size-linux-*` report, including differences from the last published
baseline. Size reports must not be placed in `dist/`.

After a successful publication, the `update-size-docs` job measures hash-verified
PyPI `tensorcodec-av` wheels for the release and pinned TorchCodec 0.17.0. It commits the published
snapshot and README table/badge with a normal push to `main`. Only this documentation
job has `contents: write`; the PyPI publisher retains OIDC plus read access. The
repository must permit the Actions bot to push these documentation updates.

If the documentation job fails after a successful publication, recover by running
`scripts/update_size_comparison.py --version <published-version>` and committing
its two outputs. Never retry publication of an already uploaded version. See the
[package size policy](package_size.md) for measurement definitions and the
[external FFmpeg guide](system_ffmpeg.md) for the optional source-build path.
