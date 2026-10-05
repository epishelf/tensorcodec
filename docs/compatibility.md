# TensorCodec compatibility contract

Reference: **TorchCodec 0.17.0**, CPU audio/video decoding. Python operations,
frame selection, ordering, timestamps, durations, stream selection and metadata
are tested independently and against this pinned version. Native decoding uses
FFmpeg; arrays are returned as NumPy instead of torch.Tensor.

## Public surface

- `tensorcodec.Frame`, `FrameBatch`, `AudioSamples`
- `tensorcodec.decoders.VideoDecoder`: constructor, `len`, integer/slice indexing,
  `get_frame_at`, `get_frames_at`, `get_frames_in_range`, `get_frame_played_at`,
  `get_frames_played_at`, `get_frames_played_in_range`, metadata, stream_index.
  Also `get_all_frames`, FPS resampling and custom JSON frame mappings.
- `tensorcodec.decoders.AudioDecoder`: constructor, metadata, stream_index,
  `get_all_samples`, `get_samples_played_in_range`, resampling and channel mixing.
- NumPy uint8, uint16 or float32 video and float32 audio; float64 batch timestamps/durations.
  Float32 RGB uses 16-bit color conversion rather than scaling uint8 output.
- NCHW/NHWC, paths/URLs, encoded bytes, uint8 arrays and seekable file-like input.
- Exact and approximate video seeking; exact is the default.
- Decoder transforms (`tensorcodec.transforms`): `Resize` (bilinear, antialiased), `CenterCrop`
  and `RandomCrop`, applied in order after display rotation, as TorchCodec does. Crops match
  TorchCodec 0.17.0 within one level; resizing is done differently and does not (see
  [resizing](#resizing)). The TorchCodec and TorchVision v2 counterparts are accepted and
  converted; other transforms fail explicitly. `RandomCrop` draws its position once per
  decoder from NumPy's global random state. Native output takes no transforms.

CUDA, torch inputs, HDR tone mapping, arbitrary-angle rotation,
video/audio encoders and samplers are outside the
CPU decoding contract. Unsupported device/transform options fail explicitly.
Do not advertise full-package or torch.Tensor type compatibility.

Image decoding and JPEG/PNG encoding have a separate [image API contract](images.md),
including supported formats, array layouts and compatibility limits.

## Playback rules

Timestamp retrieval selects frame i for `pts[i] <= t < pts[i+1]`; the last frame
ends at the content-derived stream end. This does not imply that returned frame
duration equals the next PTS gap. In 0.17.0 the implementation of time ranges
includes the frame playing at start, even when its PTS is before start. It excludes
the frame starting exactly at stop. This differs from the reference docstring;
we follow the tested implementation, with empty output when start equals stop.
Indices preserve input order and duplicates. Shape/layout, bounds, defaults and
exception classes follow the reference, including its restrictions on slice steps.

## Tests first

Fixtures are generated with FFmpeg from known grayscale frame identities and
explicit timestamp schedules, and with Python's wave module from known PCM.
ffprobe validates encoded packet metadata independently. Run the same contract
tests against the reference before adding implementation:

```sh
uv run --group oracle pytest tests/test_video_contract.py tests/test_audio_contract.py --backend torchcodec
uv run --group oracle pytest --compare
```

Comparisons check timing separately from pixels. One uint8 RGB unit or 1/65535
for float32 RGB is allowed for conversion rounding (resized frames are compared with a float64
reference instead; see [resizing](#resizing)); frame identities have independent
expectations. Tests requiring the oracle must fail on a missing/wrong reference
when `--compare` is requested. Production installation does not require torch.

Intentional fix: TensorCodec accepts empty index lists. TorchCodec 0.17.0 infers
float for empty index lists; oracle tests use explicitly typed input tensors to
isolate playback semantics from that conversion bug.

## Video fidelity extensions

PQ/HLG inputs decode as transfer-encoded RGB without tone mapping. `auto` uses
float32 for source component depths above 8 bits, including high-depth SDR.
Explicit uint16 returns full-range RGB48 and is a TensorCodec extension; it is
not native YUV output. Source `bit_depth` and `color_range` metadata are also
TensorCodec extensions. Right-angle display rotations are applied automatically;
metadata dimensions describe the rotated output. Reflected and non-right-angle
display matrices remain unsupported. Color metadata and pixel aspect ratio
describe the source; HDR output is not linear light or sRGB.

## Native video output

`output_format="native"` bypasses color conversion and display transforms for
`gray`, `gray12le`, `gray16le`, `gray16be`, `rgb24` and `rgba`. NCHW/NHWC keeps
1, 3 or 4 channels. Output uses host-endian uint8/uint16 without range scaling;
`output_dtype` may be omitted, `"auto"`, or the matching integer dtype.
`expected_pixel_format` asserts the source layout. Unsupported formats, dtype
conversions and changes of pixel format or dimensions fail explicitly.

Frame/FrameBatch `pixel_format` records the native source format and survives
indexing and FPS resampling. RGB results retain their existing behavior. Native
mode preserves encoded pixel coordinates, including inputs with display matrices.
Playback selection follows the same TorchCodec contract as RGB, including the
frame overlapping a range's start; it does not copy PyAV's legacy PTS-only range rule.

## Resizing

`Resize` resizes in YUV while converting to RGB: one bilinear swscale pass from the decoded planes
to RGB at the output size, interpolating chroma at full resolution with swscale's default
centered siting. TorchCodec 0.17.0 converts to RGB at full size, then resizes the RGB frame. In
TensorCodec the single pass costs about as much as native-size output (see the
[benchmarks](../benchmarks/README.md#decoder-transforms)).

In a pipeline:

- Crops before the first resize select its region in the decoded planes. An offset inside a
  chroma pair (an odd offset for 4:2:0) starts the chroma plane at that pair and shifts the
  chroma siting to match, so colors stay in place.
- A display rotation is applied after the first resize, to its RGB output. Right-angle rotations
  commute with the resampling, so this matches rotating first up to rounding.
- Transforms after the first resize work on its RGB output. A later resize is the same kind of
  pass, from that RGB image.
- A resize to the current size does nothing.
- After a crop, pixel formats that plane offsets cannot address (paletted, bitstream, hardware,
  or packed with subsampled chroma) are converted at full size first, and the resize then
  works from RGB.
- swscale treats a region of odd width or height as if its subsampled chroma covered the
  region exactly, which stretches chroma by up to half a pixel at the far edge. Even-sized
  regions are unaffected. With an odd offset and an even size, the last row or column has no
  chroma sample of its own beyond the region and reuses its neighbor's.

Outputs therefore differ from TorchCodec's by about 0.9 levels on average. Most of that is
TorchCodec's darker bias: swscale's fast full-size YUV to RGB conversion comes out 0.6-1.0
levels darker than exact arithmetic, and resizing afterwards keeps the bias. Native-size
decoding uses that same conversion in both libraries, so the bias is shared there. The single
pass has no such bias and is closer to a float64 reference: exact limited-range YUV to RGB with
bilinear chroma upsampling, then the antialiased bilinear filter of TorchVision v2 and PIL.

Measured on 640x480 4:2:0 clips, 20 frames each. "real" is a 10 s excerpt of the Sintel 480p
trailer cropped to 640x480; "syn" is FFmpeg `testsrc2`.

| clip | output | vs TorchCodec: max / mean / bias / >1 level / p99.9 | vs float64 reference, mean abs (bias): TorchCodec / TensorCodec |
| --- | --- | --- | --- |
| syn AV1 | 224² | 42 / 0.80 / +0.60 / 14.6% / 18 | 1.27 (−0.65) / 1.03 (−0.05) |
| syn AV1 | 128² | 25 / 0.76 / +0.60 / 15.0% / 12 | 0.84 (−0.65) / 0.60 (−0.04) |
| syn AV1 | crop 400² → 224² | 59 / 1.07 / +0.72 / 22.3% / 33 | 1.43 (−0.76) / 1.15 (−0.04) |
| real AV1 | 224² | 17 / 0.90 / +0.87 / 21.8% / 5 | 0.91 (−0.90) / 0.25 (−0.03) |
| real AV1 | 128² | 11 / 0.89 / +0.87 / 20.8% / 4 | 0.91 (−0.90) / 0.23 (−0.04) |
| real AV1 | crop 400² → 224² | 18 / 1.08 / +1.04 / 26.5% / 4 | 1.09 (−1.08) / 0.29 (−0.03) |
| syn H.264 | 224² | 49 / 0.81 / +0.59 / 15.2% / 19 | 1.27 (−0.65) / 1.05 (−0.06) |
| syn H.264 | 128² | 27 / 0.78 / +0.60 / 15.7% / 12 | 0.83 (−0.64) / 0.61 (−0.04) |
| syn H.264 | crop 400² → 224² | 58 / 1.08 / +0.72 / 22.6% / 33 | 1.43 (−0.76) / 1.16 (−0.04) |
| real H.264 | 224² | 26 / 0.90 / +0.86 / 21.9% / 5 | 0.90 (−0.89) / 0.25 (−0.03) |
| real H.264 | 128² | 16 / 0.89 / +0.86 / 20.9% / 4 | 0.90 (−0.89) / 0.22 (−0.04) |
| real H.264 | crop 400² → 224² | 27 / 1.08 / +1.04 / 26.6% / 5 | 1.08 (−1.07) / 0.29 (−0.04) |

The largest differences are at sharp, saturated color edges, where TorchCodec's
nearest-neighbor chroma followed by an RGB resize and direct bilinear chroma interpolation
disagree. Tests compare resized frames with the float64 reference on natural content: the mean
error must stay within 0.25 levels of zero and the mean absolute error below 0.75 levels. The
measured values are |bias| <= 0.16 and 0.19-0.63, and TorchCodec's path would fail the bias bound.

## Timestamp mode

`seek_mode="timestamp"` is an opt-in TensorCodec extension. Existing `exact` and
`approximate` modes are unchanged. It skips the initial full packet scan and
selects frames by `PTS[i] <= t < PTS[i+1]`, without average-FPS conversion.
The final frame requires a positive decoded duration; requests beyond that
duration, before the first frame, or with nonfinite times fail explicitly.
Missing/non-increasing decoded timestamps also fail rather than guessing.

Queries are sorted and deduplicated, then returned in the caller's order. Nearby
queries share decoding and one-frame lookahead; known later container keyframes
allow seeking across gaps. Overshooting seeks retry at exponentially earlier
positions, down to the stream start. If no frame at/before the request can be
recovered, decoding fails. Bad or absent indexes can still require substantial I/O.

Frame indices, slices, `len`, `get_all_frames`, custom mappings and ranges without
explicit `fps` are unsupported. FPS ranges retain the regular mode's resampling
timestamps and durations. Header timing remains advisory; `metadata.num_frames`
and content-derived metadata are `None`. No complete frame index is implied.
RGB/native formats, dtype, rotation, array ownership and file-like input follow
the existing output contracts. Decoder construction may still probe media data.
