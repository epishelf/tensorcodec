# Publishing TensorCodec

Release version: `0.2.0`. One release publishes two PyPI projects from the same
commit and version:

| Project | Contents | Build | Distributions |
| --- | --- | --- | --- |
| `tensorcodec` | Python API, image codecs (import `tensorcodec`) | hatchling (`uv build`) | `py3-none-any` wheel, sdist |
| `tensorcodec-native` | Rust/FFmpeg extension (import `tensorcodec_native`) | maturin (`native/`) | platform wheels, sdist |

`tensorcodec` requires `tensorcodec-native==<its own version>` behind an environment
marker matching the native wheel tags, and the `native` extra requests it
unconditionally. `tests/test_versions.py` fails CI unless `pyproject.toml`,
`native/Cargo.toml` (the native version source), both pins and
`tensorcodec.__version__` agree. At runtime `VideoDecoder`/`AudioDecoder` reject a
`tensorcodec-native` whose version differs.

Native wheels target Linux x86_64 and ARM64 (aarch64), glibc 2.17+, CPython 3.10+
(abi3). NumPy must also provide a compatible wheel for the selected Python/glibc
pair. The wheel bundles shared FFmpeg 7.1.5 and OpenSSL 3.5.9 LTS; its only Python
runtime dependency is NumPy. macOS 14+ ARM64 wheels bundle the same minimal
runtime. Windows wheels are not provided; there, `tensorcodec` provides image
codecs only. Bundled-library notices ship only with `tensorcodec-native`
(`native/licenses/`).

To release, set the new version in `pyproject.toml` (project version and both
`tensorcodec-native` pins), `native/Cargo.toml` (then `cargo update -p
tensorcodec-native --manifest-path native/Cargo.toml` to refresh the lock file)
and `src/tensorcodec/__init__.py`.

## Trusted publisher configuration

The PyPI project is already registered. Its GitHub Trusted Publisher uses:

| Field | Value |
| --- | --- |
| PyPI project names | `tensorcodec`, `tensorcodec-native` |
| GitHub owner | `epishelf` |
| Repository | `tensorcodec` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

Both projects need this publisher. `tensorcodec-native` does not exist on PyPI yet:
before its first release, add it as a pending publisher (PyPI account → Publishing)
with the same fields. Manage this configuration in each project's PyPI Publishing settings when moving
or renaming the repository or workflow. Publishing uses GitHub OIDC; no API token
is needed. Repository visibility does not need to change for a release.

## Release

Run the **Publish to PyPI** workflow on `main`. It builds the portable Linux/macOS
`tensorcodec-native` wheels and sdist plus the `tensorcodec` wheel and sdist, checks package metadata, validates the pinned oracle
and compares playback before uploading through PyPI Trusted Publishing. It uses
the existing GitHub `pypi` environment. `tensorcodec-native` is uploaded first, so the
exact pin in `tensorcodec` never points at a missing release; if the second upload
fails, rerun only it (`twine upload` of the `pypi-distributions-pure` artifact) rather
than the whole workflow. Publication fails if authorization is missing, tests fail,
or the version has already been uploaded.

```sh
gh workflow run publish.yml --repo epishelf/tensorcodec --ref main
```

For a build and full validation without uploading, pass `--field publish=false`.

Check the workflow, https://pypi.org/project/tensorcodec/ and
https://pypi.org/project/tensorcodec-native/ before reporting success. Verify a fresh
`uv pip install tensorcodec==<version>` pulls the matching `tensorcodec-native` and
decodes video without Torch/PyAV on both Linux architectures and macOS arm64, and
that it installs without `tensorcodec-native` on Windows. Update the version before subsequent releases;
PyPI versions cannot be overwritten.

The local Linux build is reproducible using `scripts/build_linux_wheel.sh` inside
`quay.io/pypa/manylinux2014_x86_64` or
`quay.io/pypa/manylinux2014_aarch64` with Rust, maturin, libclang, NASM and Perl.
Both native source archives are version- and checksum-pinned. Their licensing
and source links are recorded in `native/licenses/README.md`.

## CI versus release builds

- Ordinary CI uses prebuilt conda-forge FFmpeg 7.1.1 through Pixi, including its
  headers and shared libraries. It builds only the `tensorcodec-native` extension
  and installs `tensorcodec` from the checkout.
- The `tensorcodec` wheel is built once with `python -m build` and checked with
  `twine check --strict`; Windows and Intel macOS jobs install it without
  `tensorcodec-native` and run the image tests (one with OpenCV 5, one with 4.x).
- PyPI wheels use the smaller LGPL FFmpeg 7.1.5 build plus OpenSSL 3.5.9.
  Their native prefix is cached by architecture, glibc baseline and build-script
  checksums. This preserves the wheel's codec set, dependency size and licensing rather than bundling the full
  conda-forge dependency graph.
- Release validation installs each repaired `tensorcodec-native` wheel together with
  the `tensorcodec` wheel on glibc 2.17 with Python 3.10
  and 3.13 and decodes video/audio without Torch, PyAV or a system FFmpeg. Python
  3.10 also checks the minimum NumPy line (1.26.4). Native
  x86_64 and ARM64 runners also run the full pinned playback oracle comparison.
- Release validation still tests the installed repaired wheel. The fixture CLI
  can be FFmpeg 6 or 7; fixtures explicitly remove auxiliary sentinel packets.

## Size checks and published comparison

Final repaired `tensorcodec-native` wheels must stay within 15 MiB download and 35 MiB unpacked per
architecture. The build job checks `packaging/size-policy.json` and uploads a
separate `wheel-size-*` report, including differences from the last published
baseline. Size reports must not be placed in `dist/`.

After a successful publication, the `update-size-docs` job measures hash-verified
PyPI `tensorcodec-native` wheels for the release and pinned TorchCodec 0.17.0. It commits the published
snapshot and README table/badge with a normal push to `main`. Only this documentation
job has `contents: write`; the PyPI publisher retains OIDC plus read access. The
repository must permit the Actions bot to push these documentation updates.

If the documentation job fails after a successful publication, recover by running
`scripts/update_size_comparison.py --version <published-version>` and committing
its two outputs. Never retry publication of an already uploaded version. See the
[package size policy](package_size.md) for measurement definitions and the
[external FFmpeg guide](system_ffmpeg.md) for the optional source-build path.
