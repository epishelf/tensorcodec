"""Lazy access to the user's OpenCV installation."""


def opencv():
    try:
        import cv2
    except ImportError as exc:
        raise ImportError(
            "Image codecs require OpenCV >= 4.12; install tensorcodec[images] "
            "or use an existing compatible cv2 installation"
        ) from exc
    # 4.12: first PyPI wheels with GIF and AVIF decoders; 4.10/4.11 fail only those tests (docs/images.md).
    if tuple(int(part) for part in cv2.__version__.split(".")[:2]) < (4, 12):
        raise ImportError("Image codecs require OpenCV >= 4.12")
    return cv2
