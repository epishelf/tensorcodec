# Package size policy

TensorCodec keeps its Python dependencies to NumPy and OpenCV and bundles a minimal
FFmpeg/OpenSSL runtime in its default Linux `tensorcodec-av` wheels. Size limits prevent additions
from silently increasing the distributed binary footprint.

## What is measured

| Metric | Definition | Per-wheel limit |
| --- | --- | ---: |
| Download | Final `.whl` file size, in bytes | 15 MiB |
| Unpacked | Sum of ZIP entry file sizes, including bundled libraries | 35 MiB |

The sole policy file is [`packaging/size-policy.json`](../packaging/size-policy.json).
The checker uses only Python's standard library and never extracts the archive.
Unpacked size excludes filesystem allocation overhead. Both metrics exclude NumPy,
OpenCV, Python, package caches and other external dependencies; they are not total
installation sizes. Reports and the README use MiB (2^20 bytes).

Check final, repaired wheels locally:

```sh
uv run --no-project python scripts/check_wheel_size.py dist/*.whl --output reports/wheel-size.json
```

Each architecture is checked independently. Exactly reaching a limit passes;
exceeding either limit fails. `build-av.yml` runs this after `auditwheel
repair`, in CI and before uploading distributions. JSON reports are separate artifacts, not
files in `dist/`. Actions summaries include changes from the committed published
baseline.

Before changing a limit, explain the feature, the measured byte increase on both
architectures and why a smaller configuration would not provide the same behavior.
A limit change should be reviewed with the change that needs it.

## Decoder comparison and badge

[`packaging/size-baseline.json`](../packaging/size-baseline.json) records wheel URLs,
SHA-256 hashes, versions and sizes for TensorCodec, PyAV, TorchCodec and
PyTorch CPU. Artifacts come from PyPI except PyTorch, which uses the official
CPU index. All wheels support CPython 3.12 on Linux x86_64 or ARM64.

The README sums TorchCodec and PyTorch wheels. TensorCodec and PyAV include FFmpeg;
TorchCodec requires it separately. NumPy and other external dependencies are
excluded. These are package footprints, not complete environment sizes.

After publication, the update script verifies hashes, checks TensorCodec's limits,
and refreshes the table and badge. The badge shows the largest TensorCodec wheel
download; candidate builds never update it.

To reproduce or recover a documentation update after publication:

```sh
uv run --no-project python scripts/update_size_comparison.py --version 0.1.3
```

Review and commit `README.md` and `packaging/size-baseline.json` together. The script
requires both architectures to have been published and fails on ambiguous wheels,
yanked wheels, missing files or hash mismatches. If publication succeeded but the
documentation job failed, repair the documentation separately; do not republish
the same version. A concurrent change to `main` can reject the documentation push;
the workflow never force-pushes.

## Why FFmpeg stays bundled by default

Bundling the selected FFmpeg libraries provides one-step installation and fixes the
runtime ABI and codec configuration used by the release tests. A full conda-forge
FFmpeg environment can include many additional codec, graphics and system packages;
moving these outside the wheel does not necessarily reduce total installation size.

Reusing an existing shared FFmpeg 7 installation is an advanced source-build option:
[system FFmpeg guide](system_ffmpeg.md). FFmpeg CLI availability alone does not
satisfy the native library requirement.
