"""Image codecs work without tensorcodec-native; video/audio fail clearly when it is missing or skewed."""

import subprocess
import sys


def run_python(code):
    subprocess.run([sys.executable, "-c", code], check=True)


def test_native_extension_is_lazy():
    run_python(
        """
import sys
import tensorcodec
import tensorcodec.decoders
import tensorcodec.encoders
assert 'tensorcodec_native' not in sys.modules
"""
    )


def test_image_codecs_without_native_extension():
    run_python(
        """
import sys
sys.modules['tensorcodec_native'] = None
import numpy as np
import tensorcodec
from tensorcodec.decoders import (
    AudioDecoder, ImageReadMode, VideoDecoder, decode_avif, decode_gif, decode_image, decode_jpeg, decode_png,
    decode_webp,
)
from tensorcodec.encoders import JpegEncoder, PngEncoder

pixels = np.zeros((3, 4, 5), np.uint8)
pixels[:] = np.array([23, 91, 177], np.uint8)[:, None, None]
encoded = PngEncoder(pixels).to_tensor()
np.testing.assert_array_equal(decode_png(encoded), pixels)
np.testing.assert_array_equal(decode_image(encoded, mode=ImageReadMode.RGB), pixels)
assert decode_jpeg(JpegEncoder(pixels).to_tensor()).shape == pixels.shape

for decoder in (VideoDecoder, AudioDecoder):
    try:
        decoder(b'not media')
    except ImportError as exc:
        assert 'tensorcodec-native package' in str(exc), exc
    else:
        raise AssertionError(f'{decoder.__name__} worked without tensorcodec-native')
"""
    )


def test_version_skew_is_rejected():
    run_python(
        """
import sys
import types
package = types.ModuleType('tensorcodec_native')
package._native = types.SimpleNamespace(__version__='0.0.0', Decoder=None)
sys.modules['tensorcodec_native'] = package
from tensorcodec.decoders import VideoDecoder
try:
    VideoDecoder(b'not media')
except ImportError as exc:
    assert 'requires tensorcodec-native==' in str(exc) and 'found 0.0.0' in str(exc), exc
else:
    raise AssertionError('mismatched tensorcodec-native was accepted')
"""
    )
