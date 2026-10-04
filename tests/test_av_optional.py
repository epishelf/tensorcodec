"""Image codecs work without tensorcodec-av; video/audio fail clearly when it is missing or skewed."""

import subprocess
import sys


def run_python(code):
    subprocess.run([sys.executable, "-c", code], check=True)


def test_image_codecs_without_tensorcodec_av():
    run_python(
        """
import sys
import numpy as np
from tensorcodec.decoders import *
from tensorcodec.encoders import PngEncoder
assert 'tensorcodec_av' not in sys.modules
sys.modules['tensorcodec_av'] = None
pixels = np.full((3, 4, 5), 91, np.uint8)
np.testing.assert_array_equal(decode_image(PngEncoder(pixels).to_tensor(), mode=ImageReadMode.RGB), pixels)
for decoder in (VideoDecoder, AudioDecoder):
    try:
        decoder(b'not media')
    except ImportError as exc:
        assert 'require tensorcodec-av' in str(exc), exc
    else:
        raise AssertionError(f'{decoder.__name__} worked without tensorcodec-av')
"""
    )


def test_version_skew_is_rejected():
    run_python(
        """
import sys
import types
sys.modules['tensorcodec_av'] = types.SimpleNamespace(_av=types.SimpleNamespace(__version__='0.0.0'))
from tensorcodec.decoders import VideoDecoder
try:
    VideoDecoder(b'not media')
except ImportError as exc:
    assert 'requires tensorcodec-av==' in str(exc) and 'found 0.0.0' in str(exc), exc
else:
    raise AssertionError('mismatched tensorcodec-av was accepted')
"""
    )
