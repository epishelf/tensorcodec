"""Decoder transforms: geometric operations applied to RGB frames inside the decoder.

They mirror TorchCodec 0.17's ``torchcodec.transforms`` (and the complementary TorchVision v2 transforms): a
sequence of transforms is a pipeline, applied in order after display rotation. Crops select exact RGB pixels, as
TorchCodec's do. Resizing is bilinear with antialiasing (swscale's bilinear filter widens with the downscale ratio)
and, unlike TorchCodec, happens in YUV while converting to RGB; see `Resize`.
"""

from __future__ import annotations

import numbers
from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

__all__ = ["CenterCrop", "DecoderTransform", "RandomCrop", "Resize"]


def _size(name, size):
    if (
        isinstance(size, (str, bytes))
        or not isinstance(size, Sequence)
        or len(size) != 2
        or not all(isinstance(v, numbers.Integral) and v > 0 for v in size)
    ):
        raise ValueError(f"{name} size must be a (height, width) pair of positive integers, got {size!r}")
    return int(size[0]), int(size[1])


class DecoderTransform(ABC):
    """A transform the decoder applies before returning frames."""

    @abstractmethod
    def _op(self, input_dims):
        """(op, output (height, width)) for an input frame of `input_dims` (height, width); op is
        ("resize", height, width) or ("crop", top, left, height, width)."""


class Resize(DecoderTransform):
    """Resize to `size` (height, width); bilinear interpolation, antialiased.

    Frames are resized in YUV while converting to RGB (one swscale pass at the output size), reading any crops before
    the first resize from the source planes. TorchCodec converts to RGB at full size, then resizes, so outputs differ
    by about 0.9 levels on average, mostly TorchCodec's darker bias from its full-size conversion; this output is
    closer to a float64 reference. Differences reach a few levels on natural content and more at sharp, saturated
    color edges. See docs/compatibility.md.
    """

    def __init__(self, size):
        self.size = _size("Resize", size)

    def _op(self, input_dims):
        return ("resize", *self.size), self.size

    def __repr__(self):
        return f"Resize(size={self.size})"


class _Crop(DecoderTransform):
    def __init__(self, size):
        self.size = _size(type(self).__name__, size)

    def _fits(self, input_dims):
        if self.size[0] > input_dims[0] or self.size[1] > input_dims[1]:
            raise ValueError(f"{type(self).__name__} size {self.size} exceeds the frame size {tuple(input_dims)}")

    def __repr__(self):
        return f"{type(self).__name__}(size={self.size})"


class CenterCrop(_Crop):
    """Crop `size` (height, width) from the center; a half-pixel offset rounds to even, as in TorchCodec and
    TorchVision."""

    def _op(self, input_dims):
        self._fits(input_dims)
        top = round((input_dims[0] - self.size[0]) / 2)
        left = round((input_dims[1] - self.size[1]) / 2)
        return ("crop", top, left, *self.size), self.size


class RandomCrop(_Crop):
    """Crop `size` (height, width) at a random place, drawn once when the decoder opens (every frame of that decoder
    is cropped at the same place) from NumPy's global random state (``np.random.seed`` reproduces it; TorchVision
    draws from torch's)."""

    def _op(self, input_dims):
        self._fits(input_dims)
        top = int(np.random.randint(0, input_dims[0] - self.size[0] + 1))
        left = int(np.random.randint(0, input_dims[1] - self.size[1] + 1))
        return ("crop", top, left, *self.size), self.size


def _convert(transform):
    """A TensorCodec transform, from one or from its TorchCodec or TorchVision v2 counterpart."""
    if isinstance(transform, DecoderTransform):
        return transform
    module = type(transform).__module__
    name = type(transform).__name__
    if module.startswith("torchcodec.transforms") and name in ("Resize", "CenterCrop", "RandomCrop"):
        return globals()[name](transform.size)
    if module.startswith("torchvision.transforms.v2"):
        if name == "Resize":
            interpolation = getattr(transform.interpolation, "value", transform.interpolation)
            if interpolation != "bilinear":
                raise ValueError("TorchVision Resize must use bilinear interpolation")
            if transform.antialias is False:
                raise ValueError("TorchVision Resize must have antialias enabled")
            if transform.size is None or len(transform.size) != 2:
                raise ValueError("TorchVision Resize must have a (height, width) size")
            return Resize(transform.size)
        if name == "CenterCrop":
            return CenterCrop(transform.size)
        if name == "RandomCrop":
            if transform.padding is not None or transform.pad_if_needed:
                raise ValueError("TorchVision RandomCrop must not pad")
            return RandomCrop(transform.size)
    raise ValueError(f"Unsupported transform: {transform!r}")


def _pipeline(transforms, dims, turns):
    """Native operations for `transforms` on frames of display size `dims` (height, width): the display rotation
    (`turns` quarter turns counterclockwise) first when there are transforms, as TorchCodec applies it, then each crop
    and resize. Returns (ops, display size of the output)."""
    ops = [("rotate", turns, 0, 0, 0)] if transforms and turns else []
    for transform in transforms:
        (kind, *values), dims = _convert(transform)._op(dims)
        ops.append((kind, 0, 0, *values) if kind == "resize" else (kind, *values))
    return ops, dims
