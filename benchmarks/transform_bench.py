"""Per-frame cost of decoder transforms.

Times ``get_frames_at`` on one persistent decoder at native size and with ``Resize``, for one frame and for a
strided batch. Without ``--video``, it generates 640x480 GOP-2 synthetic clips (testsrc2) in AV1 and H.264. Run it
under two TensorCodec versions to compare them.

Usage::

    uv run --no-sync python -m benchmarks.transform_bench
    uv run --no-sync python -m benchmarks.transform_bench --video clip.mp4 --sizes 224 128 --batches 1 20
"""

from __future__ import annotations

import argparse
import statistics
import subprocess
import tempfile
import time
from pathlib import Path

ENCODERS = {"av1": ["-c:v", "libsvtav1", "-preset", "8"], "h264": ["-c:v", "libx264"]}


def _synthetic(directory: Path, codec: str) -> Path:
    path = directory / f"testsrc2_640x480_gop2_{codec}.mp4"
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
         "testsrc2=size=640x480:rate=30:duration=10", "-pix_fmt", "yuv420p", *ENCODERS[codec], "-g", "2", path],
        check=True,
    )  # fmt: skip
    return path


def _per_frame_ms(path: Path, pipelines: list[list], indices: list[int], runs: int) -> list[float]:
    """Median ms per returned frame for each pipeline; calls alternate between pipelines so drift in machine load
    affects them alike."""
    from tensorcodec.decoders import VideoDecoder

    decoders = [VideoDecoder(path, transforms=transforms) for transforms in pipelines]
    times = [[] for _ in decoders]
    try:
        for decoder in decoders:
            decoder.get_frames_at(indices)  # warm up
        for _ in range(runs):
            for decoder, samples in zip(decoders, times):
                start = time.perf_counter()
                decoder.get_frames_at(indices)
                samples.append(time.perf_counter() - start)
    finally:
        for decoder in decoders:
            decoder.close()
    return [statistics.median(samples) * 1e3 / len(indices) for samples in times]


def main() -> None:
    import tensorcodec
    from tensorcodec.decoders import VideoDecoder
    from tensorcodec.transforms import Resize

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--video", type=Path, nargs="*", help="clips to time (default: synthetic AV1 and H.264)")
    parser.add_argument("--sizes", type=int, nargs="*", default=[224, 128], help="square Resize outputs")
    parser.add_argument("--batches", type=int, nargs="*", default=[1, 20], help="frames per call")
    parser.add_argument("--runs", type=int, default=50)
    args = parser.parse_args()

    print(f"tensorcodec {tensorcodec.__version__}; ms per returned frame, median of {args.runs} calls")
    with tempfile.TemporaryDirectory() as directory:
        videos = args.video or [_synthetic(Path(directory), codec) for codec in ENCODERS]
        columns = ["native"] + [f"{size}²" for size in args.sizes]
        print("| clip | frames | " + " | ".join(columns) + " |")
        print("|---" * (len(columns) + 2) + "|")
        for path in videos:
            with VideoDecoder(path) as decoder:
                count = len(decoder)
            for batch in args.batches:
                # One frame from the middle; batches are strided across the clip, so each needs its own seek.
                # Clips shorter than the batch contribute all their frames.
                if batch == 1:
                    indices = [min(count // 2 + 1, count - 1)]
                else:
                    indices = list(range(0, count, max(count // batch, 1)))[:batch]
                pipelines = [[]] + [[Resize((size, size))] for size in args.sizes]
                cells = _per_frame_ms(path, pipelines, indices, args.runs)
                print(f"| {Path(path).name} | {len(indices)} | " + " | ".join(f"{cell:.2f}" for cell in cells) + " |")


if __name__ == "__main__":
    main()
