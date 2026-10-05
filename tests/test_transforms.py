import subprocess
from pathlib import Path

import numpy as np
import pytest

from tensorcodec.decoders import VideoDecoder
from tensorcodec.transforms import CenterCrop, RandomCrop, Resize
from tests.utils import as_numpy, index_input, run_ffmpeg


@pytest.fixture(scope="session")
def scene(tmp_path_factory):
    """Detailed 4:2:0 content, so resampling differences show."""
    path = tmp_path_factory.mktemp("transforms") / "scene.mp4"
    run_ffmpeg(
        "-f", "lavfi", "-i", "testsrc2=size=96x64:rate=10:duration=0.5",
        "-c:v", "libx264", "-threads", "1", "-crf", "0", "-g", "2", path,
    )  # fmt: skip
    return path


def _frames(path, transforms=(), **kwargs):
    with VideoDecoder(path, dimension_order="NHWC", transforms=transforms, **kwargs) as decoder:
        return decoder.get_frames_at([4, 0, 4]).data


# Resizes are checked against a float64 reference: exact limited-range YUV to RGB with bilinear (centered) chroma
# upsampling, then the antialiased bilinear filter of TorchVision v2 and PIL. On natural content swscale stays
# unbiased (|mean error| <= 0.16 levels; TorchCodec's convert-then-resize path is 0.6-1.1 levels darker) with a mean
# absolute error of 0.19-0.63 levels for even-sized regions, the residual being rounding plus swscale's chroma
# filter at sharp color edges. The bounds below leave room above those measurements and fail either bias.
NATURAL = Path(__file__).parent / "resources" / "nasa_13013.mp4"  # stream 0: 320x180 yuv420p, BT.709 limited
NATURAL_FRAMES = [0, 100, 200, 300]
MAX_BIAS, MAX_MEAN_ERROR = 0.25, 0.75
BT709 = (
    1.5748,
    0.187324,
    0.468124,
    1.8556,
)  # Kr*, Kgu, Kgv, Kb* for R = Y + Kr*V, G = Y - Kgu U - Kgv V, B = Y + Kb*U


def _filter(n_in, n_out, shift=0.0):
    """(n_out, n_in) antialiased bilinear weights: a triangle widened by the downscale ratio, pixel centers aligned."""
    scale = n_in / n_out
    centers = (np.arange(n_out) + 0.5) * scale + shift
    weights = np.maximum(0, 1 - np.abs(np.arange(n_in) + 0.5 - centers[:, None]) / max(scale, 1.0))
    return weights / weights.sum(axis=1, keepdims=True)


def _planes(path, indices, pixel_format="yuv420p", size=(180, 320)):
    """Decoded YUV planes of frames `indices` of stream 0, by the FFmpeg CLI."""
    out = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-fps_mode", "passthrough", "-f", "rawvideo",
         "-pix_fmt", pixel_format, "-"],
        capture_output=True, check=True,
    ).stdout  # fmt: skip
    (height, width), (sub_y, sub_x) = size, {"yuv420p": (2, 2), "yuv422p": (1, 2)}[pixel_format]
    luma, chroma = height * width, (height // sub_y) * (width // sub_x)
    frames = np.frombuffer(out, np.uint8).reshape(-1, luma + 2 * chroma)[indices].astype(np.float64)
    shape = (-1, height // sub_y, width // sub_x)
    return frames[:, :luma].reshape(-1, height, width), frames[:, luma:-chroma].reshape(shape), \
        frames[:, -chroma:].reshape(shape)  # fmt: skip


def _reference_rgb(y, u, v, chroma_shift=0.0):
    """Float RGB from limited-range BT.709 planes; `chroma_shift` (luma pixels, both axes) mis-sites chroma."""
    height, width = y.shape[1:]

    def up(plane):
        shifts = (chroma_shift * plane.shape[1] / height, chroma_shift * plane.shape[2] / width)  # in chroma samples
        rows, columns = _filter(plane.shape[1], height, shifts[0]), _filter(plane.shape[2], width, shifts[1])
        return np.einsum("hi,nij,wj->nhw", rows, plane, columns, optimize=True)

    y, u, v = (y - 16) * 255 / 219, (up(u) - 128) * 255 / 224, (up(v) - 128) * 255 / 224
    kr, kgu, kgv, kb = BT709
    return np.clip(np.stack([y + kr * v, y - kgu * u - kgv * v, y + kb * u], axis=-1), 0, 255)


def _reference_resize(rgb, size, crop=None):
    if crop:
        top, left, height, width = crop
        rgb = rgb[:, top : top + height, left : left + width]
    rows, columns = _filter(rgb.shape[1], size[0]), _filter(rgb.shape[2], size[1])
    return np.einsum("hi,nijc,wj->nhwc", rows, rgb, columns, optimize=True)


def _assert_near_reference(frames, reference):
    error = frames.astype(np.float64) - reference
    assert abs(error.mean()) < MAX_BIAS, error.mean()
    assert np.abs(error).mean() < MAX_MEAN_ERROR, np.abs(error).mean()


class _CropAt(CenterCrop):
    """A crop at a given (top, left, height, width)."""

    def __init__(self, top, left, height, width):
        super().__init__((height, width))
        self.top, self.left = top, left

    def _op(self, input_dims):
        return ("crop", self.top, self.left, *self.size), self.size


def _natural(transforms, **kwargs):
    with VideoDecoder(NATURAL, stream_index=0, dimension_order="NHWC", transforms=transforms, **kwargs) as decoder:
        return decoder.get_frames_at(NATURAL_FRAMES).data


@pytest.mark.parametrize("size", [(64, 64), (120, 200), (90, 160), (360, 640)])  # uneven, by 1.6, by 2, up by 2
def test_resize_matches_a_float64_reference(size):
    frames = _natural([Resize(size)])
    assert frames.shape == (4, *size, 3)
    _assert_near_reference(frames, _reference_resize(_reference_rgb(*_planes(NATURAL, NATURAL_FRAMES)), size))


@pytest.mark.parametrize("crop", [(40, 60, 100, 150), (41, 61, 100, 150)])
def test_resize_reads_preceding_crops_from_the_source_planes(crop):
    frames = _natural([_CropAt(*crop), Resize((48, 64))])
    planes = _planes(NATURAL, NATURAL_FRAMES)
    reference = _reference_resize(_reference_rgb(*planes), (48, 64), crop)
    _assert_near_reference(frames, reference)
    # Odd offsets keep the chroma siting: the output is clearly closer to the reference than to one with chroma half
    # a pixel off, the error an unadjusted odd offset would make.
    error = np.abs(frames - reference).mean()
    for shift in (-0.5, 0.5):
        missited = _reference_resize(_reference_rgb(*planes, chroma_shift=shift), (48, 64), crop)
        assert error < np.abs(frames - missited).mean() - 0.15


@pytest.mark.parametrize("pixel_format", ["yuv422p", "bgr0", "gray"])  # horizontal-only chroma, packed, one plane
def test_resize_crops_other_pixel_formats(tmp_path, pixel_format):
    path = tmp_path / f"{pixel_format}.mkv"
    run_ffmpeg("-i", NATURAL, "-map", "0:v:0", "-frames:v", 301, "-c:v", "ffv1", "-pix_fmt", pixel_format, path)
    if pixel_format == "yuv422p":
        rgb = _reference_rgb(*_planes(path, NATURAL_FRAMES, pixel_format))
    else:  # no chroma to place: the full-size RGB decode is exact
        with VideoDecoder(path, dimension_order="NHWC") as decoder:
            rgb = decoder.get_frames_at(NATURAL_FRAMES).data.astype(np.float64)
    for crop in [None, (41, 61, 100, 150)]:
        transforms = ([_CropAt(*crop)] if crop else []) + [Resize((48, 64))]
        with VideoDecoder(path, dimension_order="NHWC", transforms=transforms) as decoder:
            frames = decoder.get_frames_at(NATURAL_FRAMES).data
        _assert_near_reference(frames, _reference_resize(rgb, (48, 64), crop))


def test_crops_select_exact_pixels(scene):
    full = _frames(scene)
    np.testing.assert_array_equal(_frames(scene, [CenterCrop((31, 50))]), full[:, 16:47, 23:73])
    # half-pixel offsets round to even: 51 / 2 -> 26, 79 / 2 -> 40; 35 / 2 -> 18, 47 / 2 -> 24
    np.testing.assert_array_equal(_frames(scene, [CenterCrop((13, 17))]), full[:, 26:39, 40:57])
    np.testing.assert_array_equal(_frames(scene, [CenterCrop((29, 49))]), full[:, 18:47, 24:73])
    np.random.seed(3)
    top, left = np.random.randint(0, 64 - 20 + 1), np.random.randint(0, 96 - 30 + 1)
    np.random.seed(3)
    np.testing.assert_array_equal(_frames(scene, [RandomCrop((20, 30))]), full[:, top : top + 20, left : left + 30])


def test_a_pipeline_applies_in_order(scene):
    resized = _frames(scene, [Resize((32, 48))])
    np.testing.assert_array_equal(_frames(scene, [Resize((32, 48)), CenterCrop((16, 16))]), resized[:, 8:24, 16:32])
    cropped = _frames(scene, [CenterCrop((32, 48)), Resize((16, 24))])
    assert cropped.shape == (3, 16, 24, 3)
    reference = _frames(scene, [CenterCrop((32, 48))])
    assert np.abs(cropped.astype(int) - reference[:, ::2, ::2]).mean() < 12  # the same picture, smaller
    # Resizes after the first one resize its RGB output.
    twice = _frames(scene, [Resize((32, 48)), CenterCrop((16, 16)), Resize((8, 8))])
    assert twice.shape == (3, 8, 8, 3)
    region = resized[:, 8:24, 16:32]
    np.testing.assert_allclose(twice.mean(axis=(1, 2)), region.mean(axis=(1, 2)), atol=2)  # same colors, smaller
    # A resize to the current size changes nothing.
    np.testing.assert_array_equal(_frames(scene, [Resize((64, 96))]), _frames(scene))
    np.testing.assert_array_equal(
        _frames(scene, [CenterCrop((30, 50)), Resize((30, 50))]), _frames(scene)[:, 17:47, 23:73]
    )


@pytest.mark.parametrize("angle", [90, 180, -90])
def test_transforms_apply_after_display_rotation(scene, tmp_path, angle):
    path = tmp_path / "rotated.mp4"
    run_ffmpeg("-display_rotation", angle, "-i", scene, "-c", "copy", path)
    full = _frames(path)  # display orientation
    height, width = full.shape[1:3]
    np.random.seed(0)
    crops = [RandomCrop((height - 7, width - 11)) for _ in range(3)]
    for crop in crops:
        np.random.seed(1)
        top, left = np.random.randint(0, 8), np.random.randint(0, 12)
        np.random.seed(1)
        np.testing.assert_array_equal(_frames(path, [crop]), full[:, top : top + height - 7, left : left + width - 11])
    # Resizing happens before the rotation: the result is the unrotated resize, rotated. A crop leaving even margins
    # reads the same source region either way.
    turns = {90: 1, 180: 2, -90: 3}[angle]
    size, crop = (height // 2, width // 3), (height - 24, width - 34)
    resized = _frames(path, [CenterCrop(crop), Resize(size)])
    assert resized.shape == (3, *size, 3)
    swap = (lambda pair: pair[::-1]) if turns % 2 else (lambda pair: pair)
    unrotated = _frames(scene, [CenterCrop(swap(crop)), Resize(swap(size))])
    np.testing.assert_array_equal(resized, np.rot90(unrotated, turns, axes=(1, 2)))


def test_high_depth_output_resizes_at_depth():
    frames = _natural([Resize((120, 200))], output_dtype=np.uint16)
    assert frames.dtype == np.uint16 and frames.shape == (4, 120, 200, 3)
    reference = _reference_resize(_reference_rgb(*_planes(NATURAL, NATURAL_FRAMES)), (120, 200))
    _assert_near_reference(frames / 257, reference)
    floats = _natural([Resize((120, 200))], output_dtype=np.float32)
    np.testing.assert_allclose(floats, frames / np.float32(65535), atol=1e-7, rtol=0)


def test_timestamp_mode_and_empty_batches_take_the_output_size(scene):
    with VideoDecoder(scene, seek_mode="timestamp", dimension_order="NHWC", transforms=[Resize((16, 24))]) as d:
        assert d.get_frames_played_at([0.4, 0.0]).data.shape == (2, 16, 24, 3)
    with VideoDecoder(scene, transforms=[CenterCrop((10, 12))]) as d:
        assert d.get_frames_at([]).data.shape == (0, 3, 10, 12)


def test_invalid_transforms_are_refused(scene):
    for transforms in ([CenterCrop((65, 10))], [Resize((10, 10)), RandomCrop((11, 5))]):
        with pytest.raises(ValueError, match="exceeds"):
            VideoDecoder(scene, transforms=transforms)
    for size in [(0, 4), (4,), "ab", (2.0, 3)]:
        with pytest.raises(ValueError, match="size"):
            Resize(size)
    with pytest.raises(ValueError, match="RGB output"):
        VideoDecoder(scene, output_format="native", transforms=[Resize((8, 8))])


def test_torchcodec_and_torchvision_transforms_convert(scene):
    def counterpart(module, name, **fields):
        cls = type(name, (), {"__module__": module})
        obj = cls()
        obj.__dict__.update(fields)
        return obj

    expected = _frames(scene, [Resize((32, 48)), CenterCrop((16, 16))])
    tc = [counterpart("torchcodec.transforms._decoder_transforms", "Resize", size=[32, 48]),
          counterpart("torchcodec.transforms._decoder_transforms", "CenterCrop", size=[16, 16])]  # fmt: skip
    np.testing.assert_array_equal(_frames(scene, tc), expected)
    tv = counterpart("torchvision.transforms.v2._geometry", "Resize", size=[32, 48], interpolation="bilinear",
                     antialias=True)  # fmt: skip
    np.testing.assert_array_equal(_frames(scene, [tv]), _frames(scene, [Resize((32, 48))]))
    tv.interpolation = "nearest"
    with pytest.raises(ValueError, match="bilinear"):
        VideoDecoder(scene, transforms=[tv])
    v1 = counterpart("torchvision.transforms.transforms", "Resize", size=32, interpolation="bilinear", antialias=True)
    with pytest.raises(ValueError, match="Unsupported transform"):  # only v2, as in TorchCodec
        VideoDecoder(scene, transforms=[v1])


@pytest.mark.parametrize("case", ["swscale", "filtergraph", "rotated"])  # torchcodec's paths: width % 32, rotation
def test_transforms_match_torchcodec(oracle, scene, tmp_path, case):
    import torchcodec.transforms as tct

    path = scene
    if case == "filtergraph":
        path = tmp_path / "narrow.mp4"
        run_ffmpeg("-i", scene, "-vf", "crop=64:64", "-c:v", "libx264", "-crf", "0", path)
    if case == "rotated":
        path = tmp_path / "rotated.mp4"
        run_ffmpeg("-display_rotation", 90, "-i", scene, "-c", "copy", path)
    for ours, theirs in [
        ([CenterCrop((30, 40))], [tct.CenterCrop((30, 40))]),
        ([CenterCrop((13, 17))], [tct.CenterCrop((13, 17))]),
        ([CenterCrop((29, 49))], [tct.CenterCrop((29, 49))]),
        # Resizing differs by design (docs/compatibility.md); only the shapes agree.
        ([Resize((32, 48))], [tct.Resize((32, 48))]),
        ([Resize((40, 70))], [tct.Resize((40, 70))]),
        ([CenterCrop((40, 40)), Resize((20, 24))], [tct.CenterCrop((40, 40)), tct.Resize((20, 24))]),
    ]:
        actual = VideoDecoder(path, transforms=ours).get_frames_at([4, 0, 4])
        expected = oracle.VideoDecoder(path, transforms=theirs).get_frames_at(index_input(oracle, [4, 0, 4]))
        assert actual.data.shape == tuple(expected.data.shape)
        if not any(isinstance(t, Resize) for t in ours):
            np.testing.assert_allclose(actual.data, as_numpy(expected.data), atol=1, rtol=0)
