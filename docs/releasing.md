# Publishing TensorCodec

Release version: `0.2.0`. Distribution and import name: `tensorcodec`.
Binary wheels target Linux x86_64 and ARM64 (aarch64), glibc 2.17+, CPython 3.10+.
NumPy must also provide a compatible wheel for the selected Python/glibc pair.
The wheel bundles shared FFmpeg 7.1.5 and OpenSSL 3.5.9 LTS; its only Python
runtime dependency is NumPy. macOS 14+ ARM64 wheels bundle the same minimal
runtime. Windows wheels are not provided. A pure-Python `py3-none-any` wheel,
built by `scripts/build_pure_wheel.py` with the same metadata, covers every other
platform with image codecs only; installers prefer a matching native wheel.

## Trusted publisher configuration

The PyPI project is already registered. Its GitHub Trusted Publisher uses:

| Field | Value |
| --- | --- |
| PyPI project name | `tensorcodec` |
| GitHub owner | `epishelf` |
| Repository | `tensorcodec` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

Manage this configuration in the project's PyPI Publishing settings when moving
or renaming the repository or workflow. Publishing uses GitHub OIDC; no API token
is needed. Repository visibility does not need to change for a release.

## Release

Run the **Publish to PyPI** workflow on `main`. It builds the portable Linux/macOS wheels,
the pure-Python wheel and source distribution, checks package metadata, validates the pinned oracle
and compares playback before uploading through PyPI Trusted Publishing. It uses
the existing GitHub `pypi` environment. Publication fails if authorization is
missing, tests fail, or the version has already been uploaded.

```sh
gh workflow run publish.yml --repo epishelf/tensorcodec --ref main
```

For a build and full validation without uploading, pass `--field publish=false`.

Check the workflow and https://pypi.org/project/tensorcodec/0.2.0/ before reporting
success. Verify a fresh `uv pip install tensorcodec==0.2.0` and a decode without
Torch/PyAV on both architectures. Update the version before subsequent releases;
PyPI versions cannot be overwritten.

The local Linux build is reproducible using `scripts/build_linux_wheel.sh` inside
`quay.io/pypa/manylinux2014_x86_64` or
`quay.io/pypa/manylinux2014_aarch64` with Rust, maturin, libclang, NASM and Perl.
Both native source archives are version- and checksum-pinned. Their licensing
and source links are recorded in `licenses/README.md`.

## CI versus release builds

- Ordinary CI uses prebuilt conda-forge FFmpeg 7.1.1 through Pixi, including its
  headers and shared libraries. It builds only the TensorCodec extension.
- PyPI wheels use the smaller LGPL FFmpeg 7.1.5 build plus OpenSSL 3.5.9.
  Their native prefix is cached by architecture, glibc baseline and build-script
  checksums. This preserves the wheel's codec set, dependency size and licensing rather than bundling the full
  conda-forge dependency graph.
- Release validation installs each repaired wheel on glibc 2.17 with Python 3.10
  and 3.13 and decodes video/audio without Torch, PyAV or a system FFmpeg. Python
  3.10 also checks the minimum NumPy line (1.26.4). Native
  x86_64 and ARM64 runners also run the full pinned playback oracle comparison.
- Release validation still tests the installed repaired wheel. The fixture CLI
  can be FFmpeg 6 or 7; fixtures explicitly remove auxiliary sentinel packets.

## Size checks and published comparison

Final repaired wheels must stay within 15 MiB download and 35 MiB unpacked per
architecture. The build job checks `packaging/size-policy.json` and uploads a
separate `wheel-size-*` report, including differences from the last published
baseline. Size reports must not be placed in `dist/`.

After a successful publication, the `update-size-docs` job measures hash-verified
PyPI wheels for the release and pinned TorchCodec 0.17.0. It commits the published
snapshot and README table/badge with a normal push to `main`. Only this documentation
job has `contents: write`; the PyPI publisher retains OIDC plus read access. The
repository must permit the Actions bot to push these documentation updates.

If the documentation job fails after a successful publication, recover by running
`scripts/update_size_comparison.py --version <published-version>` and committing
its two outputs. Never retry publication of an already uploaded version. See the
[package size policy](package_size.md) for measurement definitions and the
[external FFmpeg guide](system_ffmpeg.md) for the optional source-build path.
