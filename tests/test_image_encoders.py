"""Image encoder round trips, stream writes and lazy OpenCV import."""

import subprocess
import sys
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from tensorcodec.decoders import decode_image
from tensorcodec.encoders import JpegEncoder, PngEncoder


@pytest.mark.parametrize("encoder", [JpegEncoder, PngEncoder])
@pytest.mark.parametrize("channels", [1, 3])
def test_encoder_outputs(encoder, channels, tmp_path):
    pixels = np.full((channels, 13, 17), 71, np.uint8)
    if channels == 3:
        pixels[:] = np.array([23, 91, 177])[:, None, None]
    obj = encoder(pixels)
    result = obj.to_tensor()
    assert result.ndim == 1 and result.dtype == np.uint8
    independent = np.array(Image.open(BytesIO(result.tobytes())))
    expected = pixels[0] if channels == 1 else pixels.transpose(1, 2, 0)
    np.testing.assert_allclose(independent.astype(int), expected.astype(int), atol=2 if encoder is JpegEncoder else 0)
    path, stream = tmp_path / "image.bin", BytesIO()
    obj.to_file(path)
    obj.to_file_like(stream)
    assert path.read_bytes() == stream.getvalue() == result.tobytes()
    np.testing.assert_allclose(decode_image(result, mode="UNCHANGED").astype(int), pixels.astype(int), atol=2)


def test_png_rgba_round_trip():
    rng = np.random.default_rng(0)
    pixels = rng.integers(0, 256, (4, 13, 17), dtype=np.uint8)
    pixels[3, :2] = 0  # fully transparent rows keep their color
    encoded = PngEncoder(pixels).to_tensor()
    np.testing.assert_array_equal(np.array(Image.open(BytesIO(encoded.tobytes()))), pixels.transpose(1, 2, 0))
    for mode in ("UNCHANGED", "RGBA"):
        np.testing.assert_array_equal(decode_image(encoded, mode=mode), pixels)
    np.testing.assert_array_equal(decode_image(encoded), pixels[:3])


def test_png_noncontiguous_input_and_partial_writes():
    pixels = np.arange(3 * 13 * 17, dtype=np.uint8).reshape(3, 13, 17)[:, ::-1, ::2]
    original = pixels.copy()
    encoder = PngEncoder(pixels)

    class Partial(BytesIO):
        def write(self, data):
            return super().write(data[:7])

    stream = Partial()
    encoder.to_file_like(stream, compression_level=9)
    np.testing.assert_array_equal(decode_image(stream.getvalue()), original)
    np.testing.assert_array_equal(pixels, original)


@pytest.mark.parametrize(
    "encoder,key,values",
    [
        (JpegEncoder, "quality", [0, 101, True, 1.5]),
        (PngEncoder, "compression_level", [-1, 10, True, 1.5]),
    ],
)
def test_encoder_parameter_errors(encoder, key, values):
    obj = encoder(np.zeros((3, 2, 2), np.uint8))
    for value in values:
        with pytest.raises(ValueError):
            obj.to_tensor(**{key: value})


@pytest.mark.parametrize(
    "img",
    [
        np.zeros((2, 2), np.uint8),
        np.zeros((2, 2, 2), np.uint8),
        np.zeros((5, 2, 2), np.uint8),
        np.zeros((3, 0, 2), np.uint8),
        np.zeros((3, 2, 2), np.uint16),
    ],
)
def test_encoder_input_errors(img):
    with pytest.raises(ValueError):
        PngEncoder(img)


def test_jpeg_rejects_alpha():
    with pytest.raises(ValueError, match="JpegEncoder needs .* 1/3 channels"):
        JpegEncoder(np.zeros((4, 2, 2), np.uint8))


def test_nonprogressing_writer():
    class Writer:
        def write(self, data):
            return None

    with pytest.raises(OSError, match="write length"):
        PngEncoder(np.zeros((3, 2, 2), np.uint8)).to_file_like(Writer())


def test_cv2_is_lazy():
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys
import tensorcodec.decoders
import tensorcodec.encoders
assert 'cv2' not in sys.modules
assert 'PIL' not in sys.modules
sys.modules['cv2'] = None
import numpy as np
try:
    tensorcodec.encoders.PngEncoder(np.zeros((3, 2, 2), np.uint8)).to_tensor()
except ImportError:
    pass
else:
    raise AssertionError('missing OpenCV was silently bypassed')
""",
        ],
        check=True,
    )


def test_no_silent_decode_fallback(monkeypatch):
    import cv2

    from tests.test_images import png_bytes

    monkeypatch.setattr(cv2, "imdecode", lambda *args: None)
    with pytest.raises(RuntimeError, match="OpenCV"):
        decode_image(png_bytes(np.zeros((2, 2, 3), np.uint8)))
