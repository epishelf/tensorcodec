from __future__ import annotations

import json
import math
import numbers
import os
from dataclasses import dataclass
from fractions import Fraction
from itertools import pairwise
from threading import RLock

import numpy as np

from tensorcodec._frame import AudioSamples, Frame, FrameBatch
from tensorcodec._metadata import AudioStreamMetadata, VideoStreamMetadata
from tensorcodec.transforms import _pipeline


def _native_decoder(*args):
    try:
        from tensorcodec._native import Decoder
    except ImportError as error:
        raise ImportError(
            "VideoDecoder and AudioDecoder require TensorCodec's native extension, which could not be loaded. "
            "Native wheels cover Linux x86_64/aarch64 (glibc 2.17+) and macOS 14+ arm64; elsewhere the "
            "pure-Python wheel provides only the image codecs."
        ) from error
    return Decoder(*args)


def _source(source):
    if isinstance(source, (str, os.PathLike)):
        return os.fspath(source)
    if isinstance(source, bytes):
        return source
    if isinstance(source, np.ndarray):
        if source.ndim != 1 or source.dtype != np.uint8:
            raise ValueError("encoded array must be a 1-dimensional uint8 array")
        return source.tobytes()
    if isinstance(source, (bytearray, memoryview)):
        return bytes(source)
    if hasattr(source, "read") and hasattr(source, "seek"):
        return source
    raise TypeError("source must be a path/URL, bytes, uint8 array or seekable file-like object")


def _vector(values, dtype):
    try:
        result = np.asarray(values, dtype=dtype)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("could not convert input to an array") from error
    if result.ndim != 1:
        raise ValueError("input must be one-dimensional")
    return result


class _Decoder:
    def close(self):
        with self._lock:
            self._native.close()
            self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()

    def _check_open(self):
        if self._closed:
            raise RuntimeError("decoder is closed")


@dataclass
class CpuFallbackStatus:
    status_known: bool = True

    def __bool__(self):
        return False

    def __str__(self):
        return "[CPU] Fallback status: No fallback required"


class VideoDecoder(_Decoder):
    """CPU video decoder with TorchCodec 0.17 playback rules and NumPy output."""

    def __init__(
        self,
        source,
        *,
        stream_index=None,
        dimension_order="NCHW",
        num_ffmpeg_threads=1,
        device=None,
        seek_mode="exact",
        transforms=None,
        output_dtype=None,
        custom_frame_mappings=None,
        output_format="rgb",
        expected_pixel_format=None,
    ):
        if dimension_order not in ("NCHW", "NHWC"):
            raise ValueError("dimension_order must be NCHW or NHWC")
        if seek_mode not in ("exact", "approximate", "timestamp"):
            raise ValueError("seek_mode must be exact, approximate or timestamp")
        if not isinstance(num_ffmpeg_threads, numbers.Integral) or num_ffmpeg_threads < 0:
            raise ValueError("num_ffmpeg_threads must be a nonnegative integer")
        if device is not None and str(device) != "cpu":
            raise NotImplementedError("TensorCodec currently supports CPU decoding")
        transforms = list(transforms or ())
        if transforms and output_format == "native":
            raise ValueError("transforms require RGB output")
        if custom_frame_mappings is not None and seek_mode != "exact":
            raise ValueError("custom_frame_mappings requires exact seeking")
        if output_format not in ("rgb", "native"):
            raise ValueError("output_format must be 'rgb' or 'native'")
        if expected_pixel_format is not None and output_format != "native":
            raise ValueError("expected_pixel_format requires native output")
        self.output_format = output_format
        requested_dtype = output_dtype
        if output_dtype is None:
            output_dtype = "auto" if output_format == "native" else np.uint8
        auto_dtype = isinstance(output_dtype, str) and output_dtype == "auto"
        if auto_dtype:
            self._dtype = np.dtype(np.uint8)
        else:
            try:
                self._dtype = np.dtype(output_dtype)
            except TypeError as error:
                raise ValueError("output_dtype must be uint8, uint16, float32 or auto") from error
            if self._dtype not in (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32)):
                raise ValueError("output_dtype must be uint8, uint16, float32 or auto")
        self._order = dimension_order
        self._seek_mode = seek_mode
        self._lock = RLock()
        self._closed = False
        self._native = _native_decoder(_source(source), "video", stream_index, int(num_ffmpeg_threads))
        try:
            header = self._native.metadata(apply_rotation=output_format != "native")
            self.stream_index = header["stream_index"]
            self._time_base = Fraction(header.pop("time_base_num"), header.pop("time_base_den"))
            container_duration = header.pop("container_duration")
            rotation = header["rotation"] or 0
            turns = round(rotation / 90)
            if not math.isclose(rotation, turns * 90, abs_tol=1e-3):
                raise NotImplementedError("only multiples of 90 degrees of display rotation are supported")
            self._rotation_turns = turns % 4
            if self._rotation_turns % 2:
                header["width"], header["height"] = header["height"], header["width"]
            self._ops, _ = _pipeline(transforms, (header["height"], header["width"]), self._rotation_turns)
            if auto_dtype:
                self._dtype = np.dtype(np.float32 if (header["bit_depth"] or 8) > 8 else np.uint8)
            if output_format == "native":
                pixel_format = header["pixel_format"]
                if expected_pixel_format is not None and pixel_format != expected_pixel_format:
                    raise ValueError(f"Expected source {expected_pixel_format}, got {pixel_format}")
                layouts = {
                    "gray": np.uint8,
                    "gray12le": np.uint16,
                    "gray16le": np.uint16,
                    "gray16be": np.uint16,
                    "rgb24": np.uint8,
                    "rgba": np.uint8,
                }
                if pixel_format not in layouts:
                    raise ValueError(f"Native output does not support pixel format {pixel_format!r}")
                native_dtype = np.dtype(layouts[pixel_format])
                if requested_dtype is not None and not auto_dtype and self._dtype != native_dtype:
                    raise ValueError(f"native {pixel_format} requires output_dtype={native_dtype.name} or auto")
                self._dtype = native_dtype
            numerator, denominator = header["pixel_aspect_ratio"]
            header["pixel_aspect_ratio"] = Fraction(numerator, denominator) if denominator else None
            self._mappings = None
            if custom_frame_mappings is not None:
                self._mappings = self._read_mappings(custom_frame_mappings)
            elif seek_mode == "exact":
                self._mappings = self._native.scan()
            self._set_metadata(header, container_duration)
        except BaseException:
            self._native.close()
            raise

    def _read_mappings(self, source):
        try:
            data = json.load(source) if hasattr(source, "read") else json.loads(source)
            frames = data["frames"]
            if not frames:
                raise ValueError("empty frame mappings")
            result = [
                (
                    int(f.get("pts", f.get("pkt_pts"))),
                    int(f.get("duration", f.get("pkt_duration"))),
                    bool(f["key_frame"]),
                )
                for f in frames
            ]
            if any(a[0] > b[0] for a, b in pairwise(result)):
                raise ValueError("frame mappings must be in PTS order")
            return result
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid custom_frame_mappings JSON") from error

    def _seconds(self, pts):
        # Match FFmpeg/TorchCodec's multiply-then-divide conversion.
        return float(pts) * self._time_base.numerator / self._time_base.denominator

    def _set_metadata(self, header, container_duration):
        if self._mappings is not None:
            if not self._mappings:
                raise ValueError("video contains no frames")
            begin_pts = min(f[0] for f in self._mappings)
            end_pts = max(f[0] + f[1] for f in self._mappings)
            begin = self._seconds(begin_pts)
            end = self._seconds(end_pts)
            duration = self._seconds(end_pts - begin_pts)
            count = len(self._mappings)
            content = (begin, end, count)
            average_fps = count / duration if duration > 0 else None
            self._pts = np.array([self._seconds(f[0]) for f in self._mappings])
            self._key_pts = []
            key = self._mappings[0][0]
            for pts, _, is_key in self._mappings:
                if is_key:
                    key = pts
                self._key_pts.append(key)
        elif self._seek_mode == "timestamp":
            begin = header["begin_stream_seconds_from_header"]
            duration = header["duration_seconds_from_header"]
            end = begin + duration if begin is not None and duration is not None else None
            count = None
            average_fps = header["average_fps_from_header"]
            content = (None, None, None)
        else:
            duration = header["duration_seconds_from_header"] or container_duration
            average_fps = header["average_fps_from_header"]
            count = header["num_frames_from_header"]
            if duration is None and count is not None and average_fps:
                duration = count / average_fps
            if count is None and duration is not None and average_fps:
                count = round(duration * average_fps)
            begin, end = 0.0, duration
            content = (None, None, None)
        if self._seek_mode != "timestamp" and (count is None or end is None or average_fps is None):
            raise ValueError("cannot determine frame count, timing or average fps")
        self.metadata = VideoStreamMetadata(
            **header,
            begin_stream_seconds_from_content=content[0],
            end_stream_seconds_from_content=content[1],
            num_frames_from_content=content[2],
            duration_seconds=duration,
            begin_stream_seconds=begin,
            end_stream_seconds=end,
            num_frames=count,
            average_fps=average_fps,
        )

    def _require_indices(self):
        if self._seek_mode == "timestamp":
            raise NotImplementedError("timestamp mode supports time queries only; use exact for frame indices")

    def __len__(self):
        self._require_indices()
        return self.metadata.num_frames

    @property
    def cpu_fallback(self):
        return CpuFallbackStatus()

    def _normalize_indices(self, indices):
        self._require_indices()
        values = _vector(indices, np.int64)
        values = np.where(values < 0, values + len(self), values)
        if np.any((values < 0) | (values >= len(self))):
            raise IndexError("frame index is outside the stream")
        return values

    def _targets(self, indices):
        if self._mappings is not None:
            return [(self._mappings[i][0], self._key_pts[i]) for i in indices]
        return [
            (
                int(
                    self.metadata.begin_stream_seconds / float(self._time_base)
                    + i / self.metadata.average_fps / float(self._time_base)
                ),
                int(
                    self.metadata.begin_stream_seconds / float(self._time_base)
                    + i / self.metadata.average_fps / float(self._time_base)
                ),
            )
            for i in indices
        ]

    def get_frames_at(self, indices):
        with self._lock:
            self._check_open()
            indices = self._normalize_indices(indices)
            data, pts, durations = self._native.decode_video(
                self._targets(indices),
                "native" if self.output_format == "native" else self._dtype.name,
                self._mappings is not None,
                self._ops,
            )
            return self._video_batch(data, pts, durations)

    def _video_batch(self, data, pts, durations):
        if self._rotation_turns and not self._ops:  # with transforms, the native pipeline rotated first
            # Copy to keep positive strides for consumers such as torch.from_numpy.
            data = np.rot90(data, self._rotation_turns, axes=(1, 2)).copy()
        if self._order == "NCHW":
            data = data.transpose(0, 3, 1, 2)
        return FrameBatch(
            data,
            np.asarray(pts, dtype=np.float64),
            np.asarray(durations, dtype=np.float64),
            self.metadata.pixel_format if self.output_format == "native" else None,
        )

    def get_frame_at(self, index):
        batch = self.get_frames_at([index])
        return Frame(batch.data[0], batch.pts_seconds[0], batch.duration_seconds[0], batch.pixel_format)

    def __getitem__(self, key):
        if isinstance(key, numbers.Integral):
            return self.get_frame_at(int(key)).data
        if isinstance(key, slice):
            start, stop, step = key.indices(len(self))
            return self.get_frames_in_range(start, stop, step).data
        raise TypeError("key must be an integer or slice")

    def get_frames_in_range(self, start, stop, step=1):
        start, stop, step = slice(start, stop, step).indices(len(self))
        if step <= 0:
            raise RuntimeError("step must be greater than zero")
        if stop < start:
            raise RuntimeError("negative output frame count")
        return self.get_frames_at(range(start, stop, step))

    def _indices_at_times(self, seconds):
        values = _vector(seconds, np.float64)
        if (
            np.any(~np.isfinite(values))
            or np.any(values < self.metadata.begin_stream_seconds)
            or np.any(values >= self.metadata.end_stream_seconds)
        ):
            raise RuntimeError("timestamp is outside the stream")
        if self._mappings is not None:
            return np.searchsorted(self._pts, values, side="right") - 1
        return np.floor((values - self.metadata.begin_stream_seconds) * self.metadata.average_fps).astype(np.int64)

    def get_frames_played_at(self, seconds):
        with self._lock:
            self._check_open()
            if self._seek_mode == "timestamp":
                values = _vector(seconds, np.float64)
                if np.any(~np.isfinite(values)):
                    raise ValueError("timestamps must be finite")
                # Sorted unique queries share native lookahead; restore order and duplicates afterwards.
                times, inverse = np.unique(values, return_inverse=True)
                data, pts, durations = self._native.decode_timestamps(
                    times.tolist(),
                    "native" if self.output_format == "native" else self._dtype.name,
                    self._ops,
                )
                if np.array_equal(times, values):
                    return self._video_batch(data, pts, durations)
                return self._video_batch(data[inverse], np.asarray(pts)[inverse], np.asarray(durations)[inverse])
            return self.get_frames_at(self._indices_at_times(seconds))

    def get_frame_played_at(self, seconds):
        if (
            self._seek_mode != "timestamp"
            and not self.metadata.begin_stream_seconds <= seconds < self.metadata.end_stream_seconds
        ):
            raise IndexError("timestamp is outside the stream")
        batch = self.get_frames_played_at([seconds])
        return Frame(batch.data[0], batch.pts_seconds[0], batch.duration_seconds[0], batch.pixel_format)

    def get_frames_played_in_range(self, start_seconds, stop_seconds, fps=None):
        with self._lock:
            self._check_open()
            if self._seek_mode == "timestamp":
                if fps is None:
                    raise NotImplementedError("timestamp range queries require fps; use exact for all frames")
                if not all(math.isfinite(v) for v in (start_seconds, stop_seconds, fps)) or fps <= 0:
                    raise ValueError("range bounds must be finite and fps must be positive")
                if stop_seconds < start_seconds:
                    raise ValueError("start_seconds must be <= stop_seconds")
                grid = start_seconds + np.arange(math.ceil((stop_seconds - start_seconds) * fps)) / fps
                batch = self.get_frames_played_at(grid)
                return FrameBatch(batch.data, grid, np.full(len(grid), 1 / fps), batch.pixel_format)
            if not start_seconds <= stop_seconds:
                raise ValueError("start_seconds must be <= stop_seconds")
            if not self.metadata.begin_stream_seconds <= start_seconds < self.metadata.end_stream_seconds:
                raise ValueError("start_seconds is outside the stream")
            if not stop_seconds <= self.metadata.end_stream_seconds:
                raise ValueError("stop_seconds is outside the stream")
            if fps is not None:
                if not math.isfinite(fps) or fps <= 0:
                    raise RuntimeError("fps must be finite and positive")
                count = math.ceil((stop_seconds - start_seconds) * fps)
                grid = start_seconds + np.arange(count, dtype=np.float64) / fps
                batch = self.get_frames_played_at(grid)
                return FrameBatch(batch.data, grid, np.full(count, 1 / fps, dtype=np.float64), batch.pixel_format)
            if start_seconds == stop_seconds:
                return self.get_frames_at([])
            if self._mappings is not None:
                start = int(np.searchsorted(self._pts, start_seconds, side="right") - 1)
                stop = int(np.searchsorted(self._pts, stop_seconds, side="left"))
            else:
                start = math.floor((start_seconds - self.metadata.begin_stream_seconds) * self.metadata.average_fps)
                stop = math.ceil((stop_seconds - self.metadata.begin_stream_seconds) * self.metadata.average_fps)
            return self.get_frames_in_range(start, stop)

    def get_all_frames(self, fps=None):
        self._require_indices()
        return self.get_frames_played_in_range(
            self.metadata.begin_stream_seconds, self.metadata.end_stream_seconds, fps
        )

    def _get_key_frame_indices(self):
        if self._mappings is None:
            raise RuntimeError("exact scan is required")
        return [i for i, frame in enumerate(self._mappings) if frame[2]]


class AudioDecoder(_Decoder):
    """CPU audio decoding, resampling and channel mixing with float32 output."""

    def __init__(self, source, *, stream_index=None, sample_rate=None, num_channels=None):
        for name, value in [("sample_rate", sample_rate), ("num_channels", num_channels)]:
            if value is not None and (not isinstance(value, numbers.Integral) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        self._lock = RLock()
        self._closed = False
        self._native = _native_decoder(_source(source), "audio", stream_index)
        try:
            header = self._native.metadata()
            header.pop("time_base_num")
            header.pop("time_base_den")
            container_duration = header.pop("container_duration")
            self.stream_index = header["stream_index"]
            self.metadata = AudioStreamMetadata(
                **header,
                duration_seconds=header["duration_seconds_from_header"] or container_duration,
                begin_stream_seconds=0.0,
            )
            self._rate = sample_rate if sample_rate is not None else self.metadata.sample_rate
            self._channels = num_channels if num_channels is not None else self.metadata.num_channels
        except BaseException:
            self._native.close()
            raise

    def get_all_samples(self):
        return self.get_samples_played_in_range()

    def get_samples_played_in_range(self, start_seconds=0.0, stop_seconds=None):
        with self._lock:
            self._check_open()
            if not math.isfinite(start_seconds) or (
                stop_seconds is not None and (not math.isfinite(stop_seconds) or not start_seconds <= stop_seconds)
            ):
                raise ValueError("invalid audio time range")
            data, first_pts = self._native.decode_audio(self._rate, self._channels, stop_seconds)
            last_pts = first_pts + data.shape[1] / self._rate
            if (
                data.shape[1] == 0
                or start_seconds >= last_pts
                or (stop_seconds is not None and stop_seconds <= first_pts)
            ):
                raise RuntimeError("no audio frames overlap the requested range")
            begin = max(0, round((start_seconds - first_pts) * self._rate))
            end = (
                data.shape[1]
                if stop_seconds is None
                else min(data.shape[1], round((stop_seconds - first_pts) * self._rate))
            )
            end = max(begin, end)
            data = data[:, begin:end]
            pts = max(first_pts, start_seconds)
            return AudioSamples(data, pts, data.shape[1] / self._rate, self._rate)
