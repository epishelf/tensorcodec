# Image codecs

Decode JPEG, PNG, WebP, GIF, AVIF and BMP into NumPy arrays, and encode grayscale or
RGB arrays as JPEG or PNG. Decoding returns CHW images or NCHW animations on CPU.

```sh
uv pip install 'tensorcodec[images]'
```

```python
from tensorcodec.decoders import decode_image, decode_jpeg
from tensorcodec.encoders import JpegEncoder, PngEncoder

rgb = decode_image('input.webp')             # uint8 CHW, RGB
frames = decode_image('animation.gif')      # NCHW for multiple frames
batch = decode_jpeg(['one.jpg', 'two.jpg'])  # list, possibly different sizes
PngEncoder(rgb).to_file('output.png', compression_level=6)
encoded = JpegEncoder(rgb).to_tensor(quality=90)  # 1-D uint8 NumPy array
```

## Installation

The image backend uses OpenCV 4.12 or newer, including 5.x. An existing compatible
`cv2` installation is sufficient; otherwise the `images` extra installs
`opencv-python-headless`. The bound is measured: the image test suite, including
the TorchCodec comparisons, passes with `opencv-python-headless` 4.12.0.88,
4.13.0.92 and 5.0.0.93. With 4.10 and 4.11 everything except GIF and AVIF passes:
4.12 is the first release whose PyPI wheels build both decoders (OpenCV added GIF
in 4.11, but its wheels report `GIF: NO` and `AVIF: NO`). The 4.8 and 4.9 wheels
do not import with NumPy 2. Use only one OpenCV wheel variant per environment.
NumPy is the only required dependency for the base package. Image dependencies
are separate from the base wheel size. Pillow is used only in tests.

Image codecs are pure Python and do not need `tensorcodec-native`, so they work
on every platform, including Windows and Intel macOS where `pip install
tensorcodec` installs no native package; constructing `VideoDecoder` or
`AudioDecoder` there raises `ImportError`.

## Contract and limits

- `decode_image`, `decode_jpeg`, `decode_png`, `decode_webp`, `decode_gif`,
  `decode_avif` accept paths, bytes, bytearray or 1-D uint8 arrays.
- `mode` accepts case-insensitive `UNCHANGED`, `GRAY`, `GRAY_ALPHA`, `RGB`
  (default), `RGB_ALPHA`/`RGBA`, or `ImageReadMode` values.
- `output_dtype` accepts uint8 (default), uint16 or `"auto"`. PNG preserves native
  8/16-bit precision with `"auto"`; explicit conversions scale the integer range.
- Multiple frames return NCHW; still images return CHW. Animated WebP keeps NCHW
  even with one frame. Frame timings and loop counts are not returned.
- JPEG/PNG/WebP EXIF orientation and AVIF primary-item rotation/mirror are applied.
  AVIF track-specific transforms are outside this adapter's contract.
- APNG, high-bit-depth AVIF, CMYK JPEG `UNCHANGED`, GPU decoding and nondefault
  AVIF `num_threads` raise explicit errors. OpenCV has no per-call AVIF thread
  setting; the adapter never modifies global OpenCV thread settings.
- AVIF color conversion follows OpenCV. Dropping alpha preserves straight RGB;
  TorchCodec 0.17.0 premultiplies AVIF RGB in that case. Pixel identity with
  TorchCodec is not promised across formats, builds or codec versions.
- `decode_image` also detects BMP (no format-specific function). 32-bit
  `BI_BITFIELDS` BMPs with a nonzero alpha mask keep alpha; other 32-bit BMPs are
  RGB, as in Pillow, because their fourth byte is padding.
- HEIC is unsupported. TIFF is not detected: OpenCV premultiplies unassociated
  alpha and drops gray+alpha samples, so lossless decoding cannot be promised.
- Format support depends on the installed OpenCV build; for example, the
  Windows `opencv-python-headless` 4.14 wheel has no AVIF decoder. Missing dependencies,
  unsupported codecs and decode failures raise; no alternate decoder is tried.
- Encoders accept nonempty CHW uint8 arrays with 1 or 3 channels. Both provide
  `to_file`, `to_file_like` and `to_tensor`; JPEG quality is 1–100 (default 75),
  PNG compression level is 0–9 (default 6). Encoded bytes need not match TorchCodec.

## Validation and performance

Tests cover known PNG samples, independent Pillow decoding of encoder outputs,
orientation, animations, malformed input and TorchCodec 0.17.0 comparisons.
Pillow is a test dependency, not a runtime backend.

On one Linux CPU, the adapter matched TorchCodec exactly for RGB output on 54
JPEG/PNG/WebP inputs: photograph, graphics and seeded noise at 224 square,
640×480 and 1920×1080. OpenCV was 4.13.0.92. Timings used one pinned CPU,
one OpenCV/Torch thread, three warmups and the median of five batches.

| Encoding | Adapter / TorchCodec latency, geometric mean |
| --- | ---: |
| JPEG 4:2:0 | 1.05 |
| JPEG 4:4:4 | 1.03 |
| Progressive JPEG | 1.02 |
| PNG | 1.28 |
| Lossy WebP | 1.07 |
| Lossless WebP | 1.11 |

Results are specific to this corpus and environment. Encoding speed has not been benchmarked.
With the development and oracle dependencies installed, run
`python benchmarks/image_codecs.py CORPUS_DIRECTORY` inside a uv virtual
environment to measure in-memory decoding and exact RGB agreement. The script
prints versions and per-file results; disk reads are outside timed sections.
