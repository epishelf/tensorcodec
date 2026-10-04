"""Lazy access to OpenCV, which is imported only when an image codec is used."""


def opencv():
    import cv2

    # 4.12: first PyPI wheels with GIF and AVIF decoders; 4.10/4.11 fail only those tests (docs/images.md).
    if tuple(int(part) for part in cv2.__version__.split(".")[:2]) < (4, 12):
        raise ImportError(f"Image codecs require OpenCV >= 4.12, found {cv2.__version__}")
    return cv2
