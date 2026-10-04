# Reusing an existing FFmpeg installation

## Default install

```sh
uv venv
uv pip install tensorcodec
```

On supported platforms this also installs the matching `tensorcodec-native`
wheel, which includes minimal shared FFmpeg 7.1.5 (with dav1d for AV1) and OpenSSL libraries.
No FFmpeg CLI, Pixi, Rust or libclang is required at runtime. This is the recommended
installation for a new environment.

## Source build with shared FFmpeg 7

Use this path when a server or container already provides compatible shared
FFmpeg libraries, or when you intentionally want the codec configuration of that
installation. The result depends on that external installation rather than the
bundled release libraries.

Requirements: supported Python, Rust/Cargo 1.88+, a C toolchain, Clang/libclang,
`pkg-config`, and FFmpeg 7 headers and shared libraries. An executable-only or
static-only FFmpeg installation is insufficient. FFmpeg 8/9 is not a supported
replacement for the current native boundary. Do not point a repaired PyPI wheel
at another FFmpeg installation by removing its bundled libraries.

For a Linux Pixi example, the prebuilt FFmpeg configuration used by ordinary CI is:

```sh
pixi global install --environment tensorcodec-ffmpeg 'ffmpeg=7.1.1=gpl_*'

# Pixi's default global environment location; adjust if PIXI_HOME is configured.
export FFMPEG_DIR="${PIXI_HOME:-$HOME/.pixi}/envs/tensorcodec-ffmpeg"
export LD_LIBRARY_PATH="$FFMPEG_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

test -f "$FFMPEG_DIR/include/libavcodec/avcodec.h"
test -f "$FFMPEG_DIR/lib/libavcodec.so.61"

uv venv
uv pip install --no-binary tensorcodec-native 'tensorcodec[native]'
```

`tensorcodec` is pure Python; only `tensorcodec-native` is built from source here.
Outside the platforms with native wheels (for example Intel macOS), the `native`
extra requests it explicitly. Releases up to 0.2.0 were a single package, built
from source with `--no-binary tensorcodec`.

The version/build constraint avoids silently selecting an incompatible FFmpeg
major. `FFMPEG_DIR` tells the source build where to find headers and libraries;
`LD_LIBRARY_PATH` tells the Linux loader where to find the libraries at runtime.
Keep the latter in your application/container environment. Adding the FFmpeg
executable to `PATH` is not enough. If libclang is outside the loader's search
paths, also set `LIBCLANG_PATH` to its library directory.

The example uses a GPL-enabled conda-forge build, unlike the minimal LGPL release
build. Its additional codecs, dependencies and licensing apply to your environment.
Review [`native/licenses/README.md`](../native/licenses/README.md) before redistributing a binary
built against a different FFmpeg configuration.

For development against the same prefix:

```sh
uv sync --group dev
uv run maturin develop -m native/Cargo.toml --locked --uv
```

Do not reuse an existing `dist/` wheel while verifying this path: it may contain the
bundled release libraries. To verify library loading for a source-built install:

```sh
uv run --no-sync python - <<'PY'
from tensorcodec.decoders import VideoDecoder
with VideoDecoder("video.mp4") as decoder:
    print(decoder.get_frame_at(0).data.shape)
PY
```

The native extension build and playback contracts run against prebuilt FFmpeg
7.1.1 in ordinary CI. The published-wheel tests separately verify decoding with
no external FFmpeg library installation. Configuration-specific outputs can differ
within the color-conversion tolerances described in the
[compatibility contract](compatibility.md).

## macOS wheels

`tensorcodec-native` macOS 14+ wheels support Apple Silicon, bundling FFmpeg/OpenSSL with `delocate`
(0.1.3 through 0.1.5 also had Intel wheels). CI tests
the installed wheels and clean Python 3.10/3.13 environments. Developers can run
`scripts/build_macos_wheel.sh` with Rust, Xcode tools, NASM, Meson, Ninja, pkg-config, coreutils,
maturin and delocate installed in their build environment.
