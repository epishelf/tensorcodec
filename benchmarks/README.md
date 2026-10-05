# TensorCodec benchmarks

These tools measure the current TensorCodec implementation and optional comparison
backends. Performance depends on the input, seek mode, thread count, storage and
cache state. Measure your workload before choosing a decoder.

## Setup

Use the [development environment](../README.md#development-and-verification).
TensorCodec-only runs need NumPy and TensorCodec; TorchCodec comparisons also need
the pinned oracle group. FFmpeg CLI with the requested encoders is required to
generate fixtures. Run from the repository root.

```sh
uv run --no-sync python -m benchmarks --list-decoders
uv run --no-sync python -m benchmarks --prepare --num-videos 4 --video-duration 10
uv run --no-sync python -m benchmarks --no-io --decoders tensorcodec \
  --runs 3 --output json --save benchmarks/results/speed.json
```

For a CPU comparison, explicitly select the same thread count and seek mode:

```sh
uv run --no-sync python -m benchmarks --no-io \
  --decoders tensorcodec 'torchcodec(seek=exact,thr=1)' \
  --runs 3 --output json --save benchmarks/results/speed.json
```

## Measurements

| Scenario | Workload | Included costs |
| --- | --- | --- |
| `temporal_window` | Random playback queries, 11 samples over a 1-second window | Decoder open, exact-mode packet scan, seeking, decoding, NumPy output |
| `sequential_range` | Decode the full video range | Decoder open, range selection, decoding, NumPy output |

The TensorCodec adapter opens and closes a decoder for each request. Results do
not represent a persistent decoder cache such as MediaRef's. Compare exact and
approximate modes separately: approximate mode does not guarantee exact VFR frame
selection. GPU comparison adapters include transfer back to CPU NumPy arrays.

For codec/container throughput experiments:

```sh
uv run --no-sync python -m benchmarks.container_bench --help
uv run --no-sync python -m benchmarks.container_bench
```

This tool generates fresh inputs with FFmpeg. Its codec matrix measures speed;
it does not assert pixel or timestamp correctness. See
[container and seek behavior](../docs/container_robustness.md) for tested guarantees.

## Decoder transforms

`benchmarks.transform_bench` times `get_frames_at` per returned frame on one persistent decoder,
at native size and with `Resize` to each square size. Without `--video` it generates 640x480
GOP-2 `testsrc2` clips in AV1 (SVT-AV1) and H.264. Calls alternate between pipelines, so changes
in machine load affect them alike. Run it under each version to compare.

```sh
uv run --no-sync python -m benchmarks.transform_bench
uv run --no-sync python -m benchmarks.transform_bench --video clip.mp4 --sizes 224 128 --batches 1 20
```

v0.4.1 (convert at full size, then resize in RGB) against v0.4.2 (resize in YUV while
converting, one swscale pass). Setup: Intel Core i7-14700K under WSL2, one core pinned with
`taskset`, other load present; conda-forge FFmpeg 7.1.1; one decoder thread. Values are ms per
returned frame, the mean of two runs of each version, each the median of 50 calls. All clips are
640x480, 30 fps, 300 frames, GOP 2: `syn` is `testsrc2`, `real` is 10 s of the Sintel 480p
trailer cropped to 640x480. The 1-frame case is the middle (odd) frame. The 20-frame case is
every 15th frame, so each needs its own seek and up to two decoded frames.

| clip | frames | native 0.4.1 | native 0.4.2 | 224² 0.4.1 | 224² 0.4.2 | 128² 0.4.1 | 128² 0.4.2 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| syn AV1 | 1 | 1.89 | 1.93 | 2.34 | 1.91 | 2.17 | 1.83 |
| syn AV1 | 20 | 1.62 | 1.62 | 2.09 | 1.60 | 1.89 | 1.52 |
| real AV1 | 1 | 2.09 | 1.92 | 2.52 | 1.89 | 2.37 | 1.81 |
| real AV1 | 20 | 1.58 | 1.52 | 2.05 | 1.48 | 1.87 | 1.40 |
| syn H.264 | 1 | 2.24 | 2.30 | 2.78 | 2.30 | 2.52 | 2.18 |
| syn H.264 | 20 | 1.75 | 1.78 | 2.23 | 1.75 | 2.05 | 1.67 |
| real H.264 | 1 | 3.10 | 3.06 | 3.58 | 3.08 | 3.37 | 2.96 |
| real H.264 | 20 | 2.03 | 2.02 | 2.49 | 1.99 | 2.31 | 1.91 |

Native-size decoding is unchanged; differences there are run-to-run noise. Decoding dominates.
On this machine, swscale's 640x480 yuv420p to rgb24 conversion takes about 0.07 ms per frame.
The v0.4.1 RGB resize added 0.56 ms (224²) or 0.38 ms (128²); the single pass takes 0.12 ms or
0.07 ms plus interleaving. A resized decode now costs the same as, or slightly less than, a
native-size one. See [resizing](../docs/compatibility.md#resizing) for how the pixels differ.

## Optional I/O and plotting

To separate opening from an 11-frame playback window, counting bytes returned by
file reads (including rereads, **not** physical disk or network traffic):

```sh
ffprobe -v error -select_streams v:0 -show_frames \
  -show_entries frame=pts,duration,key_frame -of json video.mp4 > frames.json
uv run --no-sync python -m benchmarks.open_cost video.mp4 --start 10 \
  --backends tensorcodec torchcodec --mappings frames.json
```

Both libraries scan packets to EOF on each fresh default `exact` open. Precomputed
`custom_frame_mappings` skip that scan while preserving exact selection; header
probing and window reads remain. Generate mappings once for the **same encoded
stream**, outside training. PTS alone is insufficient: durations and keyframe flags
are also required, in the stream's integer time base. The tool checks mapped
window pixels/PTS/durations against exact, and rotates mode order between trials;
it does not control OS caches. `approximate` is measured separately without an
accuracy guarantee. Container seek indexes are not generally complete frame maps
(e.g. MKV Cues); they cannot universally replace an exact scan.

FUSE measurements require Linux FUSE access, `pyfuse3` and `trio`. Install these
only in the benchmark environment, then omit `--no-io`. The runner can fall back
to speed-only results when FUSE is unavailable; check that `io_bytes` is populated
before reporting I/O figures. FUSE timing includes its own filesystem overhead.

```sh
uv run --no-sync python -m benchmarks --decoders tensorcodec \
  'torchcodec(seek=exact,thr=1)' --scenarios temporal_window \
  --runs 3 --output json --save benchmarks/results/io.json
```

Install `matplotlib` in the benchmark environment to plot measurements:

```sh
uv run --no-sync python -m benchmarks.plot_results \
  --speed benchmarks/results/speed.json --io benchmarks/results/io.json
```

## Reporting results

Generated JSON and charts stay in `benchmarks/results/`, which is gitignored.
Publish results only with the TensorCodec/TorchCodec/FFmpeg/NumPy versions,
hardware, corpus, seek/thread settings, cache conditions and commands. Identify
timeouts as partial results. Check frame selection and pixel agreement separately
using the [compatibility contract](../docs/compatibility.md); throughput alone is
not a correctness check.
