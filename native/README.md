# tensorcodec-native

The FFmpeg-backed extension behind [TensorCodec](https://github.com/MilkClouds/tensorcodec)'s
`VideoDecoder` and `AudioDecoder`. It has no public API: install `tensorcodec`, which
depends on the matching `tensorcodec-native` version on platforms with native wheels.

Wheels bundle shared FFmpeg, dav1d and OpenSSL libraries; see
[`licenses/README.md`](https://github.com/MilkClouds/tensorcodec/blob/main/native/licenses/README.md) for their licenses and sources.
