"""Lazy access to OpenCV, which is imported only when an image codec is used."""

# First PyPI wheels with GIF and AVIF decoders; they also require NumPy 2 (docs/images.md).
CODEC_MIN_VERSION = {"gif": (4, 12), "avif": (4, 12)}


def version(cv2):
    return tuple(int(part) for part in cv2.__version__.split(".")[:2])


def opencv(codec=None):
    import cv2

    # 4.10 wheels cannot decode animated WebP.
    if version(cv2) < (4, 11):
        raise ImportError(f"Image codecs require OpenCV >= 4.11, found {cv2.__version__}")
    if codec in CODEC_MIN_VERSION and version(cv2) < CODEC_MIN_VERSION[codec]:
        required = ".".join(map(str, CODEC_MIN_VERSION[codec]))
        raise RuntimeError(
            f"{codec.upper()} decoding requires OpenCV >= {required} (which requires NumPy >= 2), "
            f"found {cv2.__version__}"
        )
    return cv2
