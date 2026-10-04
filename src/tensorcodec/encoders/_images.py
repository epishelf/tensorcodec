"""TorchCodec-style image encoding delegated to OpenCV."""

from pathlib import Path

import numpy as np

from tensorcodec._opencv import opencv


class _ImageEncoder:
    _channels = (1, 3)

    def __init__(self, img):
        if not isinstance(img, np.ndarray):
            raise TypeError("img must be a NumPy array")
        if img.dtype != np.uint8 or img.ndim != 3 or img.shape[0] not in self._channels or 0 in img.shape:
            channels = "/".join(map(str, self._channels))
            raise ValueError(f"{type(self).__name__} needs a nonempty CHW uint8 array with {channels} channels")
        self.img = img

    def _encode(self, extension, parameter, value, low, high):
        if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
            raise ValueError(f"encoding parameter must be an integer in [{low}, {high}]")
        cv = opencv()
        channels = len(self.img)
        # RGB(A) to OpenCV's BGR(A)
        pixels = self.img[0] if channels == 1 else self.img.transpose(1, 2, 0)[..., [2, 1, 0, 3][:channels]]
        try:
            ok, encoded = cv.imencode(extension, np.ascontiguousarray(pixels), [getattr(cv, parameter), value])
        except cv.error as exc:
            raise RuntimeError(f"OpenCV failed to encode {extension}: {exc}") from exc
        if not ok:
            raise RuntimeError(f"OpenCV could not encode {extension}")
        return encoded.reshape(-1)

    def to_file(self, dest, **kwargs):
        """Write encoded bytes to a filesystem path."""
        Path(dest).write_bytes(self.to_tensor(**kwargs).tobytes())

    def to_file_like(self, dest, **kwargs):
        """Write encoded bytes to a binary file-like object."""
        data = self.to_tensor(**kwargs).tobytes()
        offset = 0
        while offset < len(data):
            written = dest.write(data[offset:])
            if not isinstance(written, int) or written <= 0 or written > len(data) - offset:
                raise OSError("file-like object did not report a valid write length")
            offset += written


class JpegEncoder(_ImageEncoder):
    """Encode a CHW uint8 grayscale/RGB image on CPU."""

    def to_tensor(self, *, quality=75):
        """Return encoded JPEG bytes as a one-dimensional uint8 NumPy array."""
        return self._encode(".jpg", "IMWRITE_JPEG_QUALITY", quality, 1, 100)


class PngEncoder(_ImageEncoder):
    """Encode a CHW uint8 grayscale/RGB/RGBA image on CPU."""

    _channels = (1, 3, 4)

    def to_tensor(self, *, compression_level=6):
        """Return encoded PNG bytes as a one-dimensional uint8 NumPy array."""
        return self._encode(".png", "IMWRITE_PNG_COMPRESSION", compression_level, 0, 9)
