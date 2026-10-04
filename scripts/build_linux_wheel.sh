#!/usr/bin/env bash
# Run inside manylinux2014 with Rust, maturin, libclang, NASM, Meson, Ninja and Perl installed.
set -euo pipefail
export BINDGEN_EXTRA_CLANG_ARGS="-I$(gcc -print-file-name=include)${BINDGEN_EXTRA_CLANG_ARGS:+ $BINDGEN_EXTRA_CLANG_ARGS}"
build_prefix="${TENSORCODEC_NATIVE_PREFIX:-/opt/tensorcodec}"
scripts/build_openssl.sh "$build_prefix/openssl"
export PKG_CONFIG_PATH="$build_prefix/openssl/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
export LD_LIBRARY_PATH="$build_prefix/openssl/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
scripts/build_ffmpeg.sh "$build_prefix/ffmpeg"
export FFMPEG_DIR="$build_prefix/ffmpeg"
export LD_LIBRARY_PATH="$FFMPEG_DIR/lib:$LD_LIBRARY_PATH"
maturin build -m av/Cargo.toml --release --locked --auditwheel repair --compatibility manylinux2014 --out dist
python scripts/check_wheel_size.py dist/*.whl
