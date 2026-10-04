#!/usr/bin/env bash
# Build the same minimal LGPL runtime as Linux, then relocate its dylibs.
set -euo pipefail
export PATH="$(brew --prefix coreutils)/libexec/gnubin:$PATH"
export MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-14.0}"
build_prefix="$PWD/.native-deps/macos"
export PKG_CONFIG_PATH="$build_prefix/openssl/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"
if [ ! -f "$build_prefix/ready" ]; then
  bash scripts/build_openssl.sh "$build_prefix/openssl"
  bash scripts/build_ffmpeg.sh "$build_prefix/ffmpeg"
  touch "$build_prefix/ready"
fi
export FFMPEG_DIR="$build_prefix/ffmpeg"
maturin build -m av/Cargo.toml --release --locked --out unrepaired
delocate-wheel --require-archs "$(uname -m)" -w dist unrepaired/*.whl
python scripts/check_wheel_size.py dist/*.whl
