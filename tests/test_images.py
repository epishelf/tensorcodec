"""Image API contracts from known PNG samples and independent FFmpeg encodings."""

import struct
import zlib
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tests.utils import as_numpy, run_ffmpeg


def chunk(kind, data):
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def png_bytes(pixels):
    height, width, channels = pixels.shape
    depth = pixels.dtype.itemsize * 8
    color = {1: 0, 2: 4, 3: 2, 4: 6}[channels]

    raw = pixels.astype(">u2" if depth == 16 else np.uint8).tobytes()
    stride = width * channels * (depth // 8)
    scanlines = b"".join(b"\0" + raw[i : i + stride] for i in range(0, len(raw), stride))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, depth, color, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(scanlines))
        + chunk(b"IEND", b"")
    )


def dtype_for(backend, name):
    if name == "auto":
        return name
    if backend.__name__.startswith("torchcodec"):
        import torch

        return getattr(torch, name)
    return getattr(np, name)


@pytest.fixture(scope="module")
def encoded_images(tmp_path_factory):
    root = tmp_path_factory.mktemp("images")
    pixels = np.arange(16 * 24 * 3, dtype=np.uint8).reshape(16, 24, 3)
    (root / "source.png").write_bytes(png_bytes(pixels))
    for codec, options in {
        "jpeg": ["-c:v", "mjpeg", "-q:v", "1", "-pix_fmt", "yuvj444p"],
        "gif": [],
        "avif": ["-c:v", "libaom-av1", "-still-picture", "1", "-crf", "0", "-cpu-used", "8"],
    }.items():
        run_ffmpeg("-i", root / "source.png", "-frames:v", 1, *options, "-threads", 1, root / f"image.{codec}")
    Image.fromarray(pixels).save(root / "image.webp", lossless=True)
    # Numbered sequence: FFmpeg's glob pattern type is unavailable on Windows.
    for index, frame in enumerate((pixels, 255 - pixels)):
        (root / f"frame{index}.png").write_bytes(png_bytes(frame))
    run_ffmpeg("-framerate", 2, "-i", root / "frame%d.png", "-threads", 1, root / "animated.gif")
    return root


@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("depth", [8, 16])
def test_png_native_samples(backend, channels, depth):
    dtype = np.uint8 if depth == 8 else np.uint16
    pixels = np.arange(5 * 7 * channels, dtype=dtype).reshape(5, 7, channels)
    if depth == 16:
        pixels = pixels * 311
    decoded = backend.decode_png(png_bytes(pixels), mode="UNCHANGED", output_dtype="auto")
    np.testing.assert_array_equal(as_numpy(decoded), pixels.transpose(2, 0, 1))
    assert as_numpy(decoded).dtype == dtype


@pytest.mark.parametrize("mode,channels", [("RGB", 3), ("GRAY", 1), ("RGB_ALPHA", 4), ("GRAY_ALPHA", 2)])
def test_png_color_modes(backend, mode, channels):
    pixels = np.array([[[255, 0, 0, 17], [0, 255, 0, 93], [0, 0, 255, 201]]], np.uint8)
    result = as_numpy(backend.decode_image(png_bytes(pixels), mode=mode.lower()))
    assert result.shape == (channels, 1, 3)
    if "ALPHA" in mode:
        np.testing.assert_array_equal(result[-1], pixels[..., 3])
    if mode.startswith("RGB"):
        np.testing.assert_array_equal(result[:3], pixels[..., :3].transpose(2, 0, 1))
    else:
        np.testing.assert_array_equal(result[0], [[76, 149, 29]])


def test_output_dtype_scales_values(backend):
    pixels = np.array([[[0], [1], [128], [255]]], np.uint8)
    result = backend.decode_png(png_bytes(pixels), mode="GRAY", output_dtype=dtype_for(backend, "uint16"))
    np.testing.assert_array_equal(as_numpy(result), pixels.transpose(2, 0, 1).astype(np.uint16) * 257)
    pixels = np.array([[[0], [128], [129], [32768], [65535]]], np.uint16)
    result = backend.decode_png(png_bytes(pixels), mode="GRAY")
    np.testing.assert_array_equal(as_numpy(result), np.rint(pixels.transpose(2, 0, 1) / 257).astype(np.uint8))


@pytest.mark.parametrize("kind", ["bytes", "bytearray", "path", "str", "array"])
def test_image_sources_and_content_detection(backend, tmp_path, kind):
    pixels = np.arange(27, dtype=np.uint8).reshape(3, 3, 3)
    data = png_bytes(pixels)
    path = tmp_path / "misleading.jpg"
    path.write_bytes(data)
    sources = {"bytes": data, "bytearray": bytearray(data), "path": path, "str": str(path)}
    if kind == "array":
        if backend.__name__.startswith("torchcodec"):
            import torch

            source = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        else:
            source = np.frombuffer(data, dtype=np.uint8)
    else:
        source = sources[kind]
    np.testing.assert_array_equal(as_numpy(backend.decode_image(source)), pixels.transpose(2, 0, 1))


@pytest.mark.parametrize("codec", ["jpeg", "webp", "gif", "avif"])
def test_format_functions_and_dispatch(backend, encoded_images, codec):
    path = encoded_images / f"image.{codec}"
    if codec == "avif" and not backend.__name__.startswith("torchcodec"):
        import cv2

        if not cv2.haveImageReader(str(path)):  # e.g. opencv-python-headless 4.14 on Windows
            with pytest.raises(RuntimeError, match="codec build support"):
                backend.decode_avif(path)
            return
    direct = as_numpy(getattr(backend, f"decode_{codec}")(path))
    assert direct.shape == (3, 16, 24)
    assert direct.dtype == np.uint8
    np.testing.assert_array_equal(direct, as_numpy(backend.decode_image(path.read_bytes())))


def test_jpeg_batch(backend, encoded_images):
    source = encoded_images / "image.jpeg"
    images = backend.decode_jpeg([source, source.read_bytes()])
    assert isinstance(images, list)
    assert len(images) == 2
    np.testing.assert_array_equal(as_numpy(images[0]), as_numpy(images[1]))
    images[0][...] = 0
    assert as_numpy(images[1]).any()
    assert backend.decode_jpeg([]) == []


def test_gif_animation(backend, encoded_images):
    frames = as_numpy(backend.decode_gif(encoded_images / "animated.gif"))
    assert frames.shape == (2, 3, 16, 24)
    assert not np.array_equal(frames[0], frames[1])


@pytest.mark.parametrize("mode", ["RGB", "UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB_ALPHA"])
@pytest.mark.parametrize("codec", ["jpeg", "png", "webp", "gif", "avif"])
def test_image_differential(oracle, encoded_images, codec, mode):
    import tensorcodec.decoders as actual

    path = encoded_images / ("source.png" if codec == "png" else f"image.{codec}")
    got = getattr(actual, f"decode_{codec}")(path, mode=mode)
    expected = as_numpy(getattr(oracle, f"decode_{codec}")(path, mode=mode))
    assert got.shape == expected.shape
    np.testing.assert_allclose(got.astype(np.int32), expected.astype(np.int32), atol=2, rtol=0)


def test_image_errors_and_cpu_boundary():
    from tensorcodec.decoders import decode_avif, decode_image, decode_jpeg, decode_png

    data = png_bytes(np.zeros((2, 3, 3), np.uint8))
    with pytest.raises(ValueError, match="mode"):
        decode_png(data, mode="BGR")
    with pytest.raises(ValueError, match="output_dtype"):
        decode_png(data, output_dtype=np.float32)
    with pytest.raises(ValueError, match="one-dimensional"):
        decode_png(np.zeros((2, 3), np.uint8))
    with pytest.raises(TypeError, match="source"):
        decode_image(object())
    with pytest.raises(ValueError, match="CPU"):
        decode_jpeg(b"", device="cuda")
    with pytest.raises(RuntimeError, match="expected jpeg"):
        decode_jpeg(data)
    with pytest.raises(ValueError, match="unrecognized"):
        decode_image(b"not an image")
    with pytest.raises(ValueError, match="num_threads"):
        decode_avif(b"", num_threads=0)
    with pytest.raises(FileNotFoundError):
        decode_image(Path("/nonexistent/image.png"))


def test_heic_is_explicitly_unsupported():
    from tensorcodec.decoders import decode_image

    data = struct.pack(">I", 24) + b"ftypheic" + b"\0" * 4 + b"mif1heic"
    with pytest.raises(NotImplementedError, match="HEIC"):
        decode_image(data)


@pytest.fixture(scope="module")
def optional_images(tmp_path_factory):
    root = tmp_path_factory.mktemp("optional-images")
    pixels = np.zeros((16, 24, 4), np.uint8)
    pixels[..., :3] = [30, 60, 90]
    pixels[..., 3] = 255
    pixels[:8, :12, 3] = 0
    image = Image.fromarray(pixels)
    second = image.copy()
    second.paste((120, 50, 10, 128), (8, 4, 16, 12))
    image.save(root / "animation.webp", save_all=True, append_images=[second], lossless=True, duration=100)
    return root


@pytest.mark.parametrize("mode", ["RGB", "UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB_ALPHA"])
def test_animated_webp_differential(oracle, optional_images, mode):
    from tensorcodec.decoders import decode_image

    path = optional_images / "animation.webp"
    got = decode_image(path, mode=mode)
    expected = as_numpy(oracle.decode_image(path, mode=mode))
    assert got.shape == expected.shape
    np.testing.assert_allclose(got.astype(np.int32), expected.astype(np.int32), atol=2, rtol=0)


def test_single_frame_animated_webp_keeps_batch_dimension(oracle, optional_images):
    from tensorcodec.decoders import decode_webp

    data = (optional_images / "animation.webp").read_bytes()
    chunks, offset, frames = [], 12, 0
    while offset + 8 <= len(data):
        size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        end = offset + 8 + size + (size & 1)
        if data[offset : offset + 4] == b"ANMF":
            frames += 1
        if frames < 2:
            chunks.append(data[offset:end])
        offset = end
    body = b"WEBP" + b"".join(chunks)
    encoded = b"RIFF" + struct.pack("<I", len(body)) + body
    actual, expected = decode_webp(encoded), as_numpy(oracle.decode_webp(encoded))
    assert actual.shape == (1, 3, 16, 24)
    np.testing.assert_array_equal(actual, expected)


def test_transparent_gif_matches_oracle(oracle, tmp_path):
    from tensorcodec.decoders import decode_gif

    image = Image.new("P", (16, 16))
    image.putpalette([90, 30, 10, 10, 150, 30] + [0] * 762)
    image.paste(1, (4, 4, 12, 12))
    path = tmp_path / "transparent.gif"
    image.save(path, transparency=0, background=0)
    for mode in ("RGB", "UNCHANGED", "GRAY_ALPHA"):
        actual = decode_gif(path, mode=mode)
        expected = as_numpy(oracle.decode_gif(path, mode=mode))
        assert actual.shape == expected.shape
        if actual.shape[0] in (2, 4):
            np.testing.assert_array_equal(actual[-1], expected[-1])
            opaque = actual[-1] != 0
            np.testing.assert_allclose(actual[:-1, opaque], expected[:-1, opaque], atol=1, rtol=0)
        else:
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("disposal", [1, 2, 3])
def test_gif_disposal(oracle, tmp_path, disposal):
    from tensorcodec.decoders import decode_gif

    frames = []
    for index in range(3):
        image = Image.new("P", (16, 16))
        image.putpalette([90, 30, 10, 10, 150, 30] + [0] * 762)
        image.paste(1, (index * 4, index * 4, index * 4 + 4, index * 4 + 4))
        frames.append(image)
    path = tmp_path / "disposal.gif"
    frames[0].save(path, save_all=True, append_images=frames[1:], transparency=0, background=0, disposal=disposal)
    for mode in ("RGB", "RGB_ALPHA"):
        actual, expected = decode_gif(path, mode=mode), as_numpy(oracle.decode_gif(path, mode=mode))
        if mode == "RGB_ALPHA":
            np.testing.assert_array_equal(actual[:, 3], expected[:, 3])
            mask = actual[:, 3] != 0
            actual, expected = actual.transpose(0, 2, 3, 1)[mask], expected.transpose(0, 2, 3, 1)[mask]
        np.testing.assert_array_equal(actual, expected)


def test_high_depth_avif(oracle, tmp_path):
    from tensorcodec.decoders import decode_avif

    pixels = np.arange(16 * 24 * 3, dtype=np.uint16).reshape(16, 24, 3) * 53
    source, path = tmp_path / "high.png", tmp_path / "high.avif"
    source.write_bytes(png_bytes(pixels))
    run_ffmpeg(
        "-i",
        source,
        "-frames:v",
        1,
        "-c:v",
        "libaom-av1",
        "-pix_fmt",
        "yuv444p10le",
        "-still-picture",
        1,
        "-crf",
        0,
        "-cpu-used",
        8,
        "-threads",
        1,
        path,
    )
    with pytest.raises(NotImplementedError, match="high-bit-depth"):
        decode_avif(path, output_dtype="auto")


def test_corrupt_png_raises():
    from tensorcodec.decoders import decode_png

    data = png_bytes(np.zeros((8, 8, 3), np.uint8))
    with pytest.raises(RuntimeError):
        decode_png(data[:45])


def test_cmyk_jpeg_modes(oracle):
    from tensorcodec.decoders import decode_jpeg

    image = Image.new("CMYK", (8, 8), (30, 50, 70, 90))
    encoded = BytesIO()
    image.save(encoded, format="JPEG")
    data = encoded.getvalue()
    for mode in ("RGB", "GRAY", "RGB_ALPHA", "GRAY_ALPHA"):
        actual, expected = decode_jpeg(data, mode=mode), as_numpy(oracle.decode_jpeg(data, mode=mode))
        np.testing.assert_allclose(actual.astype(int), expected.astype(int), atol=2, rtol=0)
        if "ALPHA" in mode:
            assert (actual[-1] == 255).all()
    with pytest.raises(NotImplementedError, match="CMYK"):
        decode_jpeg(data, mode="UNCHANGED")


@pytest.mark.parametrize("codec", ["PNG", "JPEG", "WEBP", "AVIF"])
@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_orientation(oracle, codec, orientation):
    from tensorcodec.decoders import decode_image

    pixels = np.arange(16 * 24 * 3, dtype=np.uint8).reshape(16, 24, 3)
    image, output = Image.fromarray(pixels), BytesIO()
    exif = Image.Exif()
    exif[274] = orientation
    image.save(
        output, format=codec, exif=exif, quality=100, subsampling="4:4:4" if codec == "AVIF" else 0, max_threads=1
    )
    actual, expected = decode_image(output.getvalue()), as_numpy(oracle.decode_image(output.getvalue()))
    assert actual.shape == expected.shape
    np.testing.assert_allclose(actual.astype(int), expected.astype(int), atol=2, rtol=0)


@pytest.mark.parametrize("depth", [8, 16])
@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_png_all_modes_exact(oracle, depth, channels):
    from tensorcodec.decoders import decode_png

    dtype = np.uint8 if depth == 8 else np.uint16
    pixels = np.random.default_rng(42).integers(0, 2**depth, (29, 37, channels), dtype=dtype)
    data = png_bytes(pixels)
    for mode in ("UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA"):
        for output in ("auto", "uint8", "uint16"):
            actual = decode_png(data, mode=mode, output_dtype=output)
            expected = as_numpy(oracle.decode_png(data, mode=mode, output_dtype=dtype_for(oracle, output)))
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("codec", ["JPEG", "PNG", "WEBP"])
@pytest.mark.parametrize("content", ["noise", "graphics"])
def test_textured_images_exact(oracle, codec, content):
    from tensorcodec.decoders import decode_image

    if content == "noise":
        pixels = np.random.default_rng(42).integers(0, 256, (127, 193, 3), dtype=np.uint8)
    else:
        y, x = np.indices((127, 193))
        pixels = np.stack(((x // 17 % 2) * 255, (y // 23 % 2) * 255, ((x + y) // 31 % 2) * 255), axis=-1)
        pixels = pixels.astype(np.uint8)
    for options in ({"subsampling": 2}, {"subsampling": 0}, {"progressive": True}) if codec == "JPEG" else ({},):
        output = BytesIO()
        Image.fromarray(pixels).save(output, format=codec, quality=90, **options)
        data = output.getvalue()
        np.testing.assert_array_equal(decode_image(data), as_numpy(oracle.decode_image(data)))


@pytest.mark.parametrize("mode", ["P", "L", "RGB"])
def test_png_transparency_key(oracle, mode):
    from tensorcodec.decoders import decode_png

    image = Image.new(mode, (19, 13))
    if mode == "P":
        image.putpalette([10, 20, 30, 50, 60, 70] + [0] * 762)
        image.paste(1, (0, 0, 8, 9))
    else:
        image.paste(123 if mode == "L" else (10, 20, 30), (0, 0, 8, 9))
    output = BytesIO()
    image.save(output, format="PNG", transparency=(10, 20, 30) if mode == "RGB" else 0)
    for read_mode in ("UNCHANGED", "GRAY", "GRAY_ALPHA", "RGB", "RGB_ALPHA"):
        for dtype in ("auto", "uint8", "uint16"):
            data = output.getvalue()
            actual = decode_png(data, mode=read_mode, output_dtype=dtype)
            expected = as_numpy(oracle.decode_png(data, mode=read_mode, output_dtype=dtype_for(oracle, dtype)))
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("codec", ["JPEG", "PNG", "WEBP", "GIF", "AVIF"])
def test_truncated_images_raise(codec):
    from tensorcodec.decoders import decode_image

    output = BytesIO()
    Image.new("RGB", (37, 29), (30, 60, 90)).save(output, format=codec, max_threads=1)
    data = output.getvalue()
    with pytest.raises((RuntimeError, ValueError)):
        decode_image(data[: len(data) // 2])


def test_avif_alpha(oracle):
    from tensorcodec.decoders import decode_avif

    pixels = np.random.default_rng(42).integers(0, 256, (31, 43, 4), dtype=np.uint8)
    output = BytesIO()
    Image.fromarray(pixels).save(output, format="AVIF", quality=90, subsampling="4:4:4", max_threads=1)
    # OpenCV preserves straight RGB; TorchCodec premultiplies when dropping AVIF alpha.
    data = output.getvalue()
    rgba = decode_avif(data, mode="RGB_ALPHA")
    expected = as_numpy(oracle.decode_avif(data, mode="RGB_ALPHA"))
    np.testing.assert_allclose(rgba.astype(int), expected.astype(int), atol=1, rtol=0)
    np.testing.assert_array_equal(decode_avif(data), rgba[:3])


def test_png_adam7_interlace(oracle):
    from tensorcodec.decoders import decode_png

    pixels = np.random.default_rng(42).integers(0, 65536, (29, 37, 4), dtype=np.uint16)
    raw = bytearray()
    for x, y, dx, dy in (
        (0, 0, 8, 8),
        (4, 0, 8, 8),
        (0, 4, 4, 8),
        (2, 0, 4, 4),
        (0, 2, 2, 4),
        (1, 0, 2, 2),
        (0, 1, 1, 2),
    ):
        for row in pixels[y::dy, x::dx]:
            raw.extend(b"\0" + row.astype(">u2").tobytes())
    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 37, 29, 16, 6, 0, 0, 1))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )
    for mode in ("UNCHANGED", "RGB", "RGB_ALPHA", "GRAY", "GRAY_ALPHA"):
        actual = decode_png(data, mode=mode, output_dtype="auto")
        expected = as_numpy(oracle.decode_png(data, mode=mode, output_dtype="auto"))
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_low_bit_palette_png(oracle, bits):
    from tensorcodec.decoders import decode_png

    image = Image.new("P", (37, 29))
    image.putpalette(list(range(256)) * 3)
    image.putdata((np.arange(37 * 29) % (2**bits)).tolist())
    buffer = BytesIO()
    image.save(buffer, format="PNG", bits=bits, transparency=0)
    for mode in ("UNCHANGED", "RGB", "RGB_ALPHA", "GRAY", "GRAY_ALPHA"):
        np.testing.assert_array_equal(
            decode_png(buffer.getvalue(), mode=mode), as_numpy(oracle.decode_png(buffer.getvalue(), mode=mode))
        )


@pytest.mark.parametrize("codec", ["WEBP", "GIF", "AVIF"])
def test_image_animation_all_frames(oracle, codec):
    from tensorcodec.decoders import decode_image

    rng = np.random.default_rng(42)
    frames = [Image.fromarray(rng.integers(0, 256, (29, 37, 3), dtype=np.uint8)) for _ in range(3)]
    output = BytesIO()
    frames[0].save(output, format=codec, save_all=True, append_images=frames[1:], duration=100, max_threads=1)
    actual = decode_image(output.getvalue())
    assert actual.shape == (3, 3, 29, 37)
    if codec == "AVIF":
        # AVIF YUV conversion belongs to the installed OpenCV/libavif build.
        import cv2

        ok, decoded = cv2.imdecodemulti(
            np.frombuffer(output.getvalue(), np.uint8).reshape(1, -1), cv2.IMREAD_UNCHANGED
        )
        assert ok
        expected = np.stack([frame[..., ::-1].transpose(2, 0, 1) for frame in decoded])
    else:
        expected = as_numpy(oracle.decode_image(output.getvalue()))
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("bits", [1, 2, 4])
def test_low_bit_grayscale_transparency(bits):
    from tensorcodec.decoders import decode_png

    data = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 1, bits, 0, 0, 0, 0))
        + chunk(b"tRNS", struct.pack(">H", 1))
        + chunk(b"IDAT", zlib.compress(bytes([0, 1 << (8 - bits)])))
        + chunk(b"IEND", b"")
    )
    gray = 255 // ((1 << bits) - 1)
    for mode, channels in (("GRAY_ALPHA", 1), ("RGB_ALPHA", 3)):
        result = decode_png(data, mode=mode)
        np.testing.assert_array_equal(result[:-1], np.tile([[[gray, 0]]], (channels, 1, 1)))
        np.testing.assert_array_equal(result[-1], [[0, 255]])


def bitfields_bmp(rgba, header_size, alpha_mask):
    """32-bit BI_BITFIELDS BMP; header_size 40 stores RGB masks only, 124 is BITMAPV5HEADER."""
    height, width, _ = rgba.shape
    pixels = rgba[::-1, :, [2, 1, 0, 3]].tobytes()
    header = struct.pack("<IiiHHIIiiII", header_size, width, height, 1, 32, 3, len(pixels), 2835, 2835, 0, 0)
    masks = struct.pack("<IIII", 0x00FF0000, 0x0000FF00, 0x000000FF, alpha_mask)
    if header_size == 40:
        header += masks[:12]
    else:
        header += masks + b"BGRs" + bytes(48) + struct.pack("<IIII", 4, 0, 0, 0)
    offset = 14 + len(header)
    return b"BM" + struct.pack("<IHHI", offset + len(pixels), 0, 0, offset) + header + pixels


@pytest.mark.parametrize(("pil_mode", "channels"), [("L", 1), ("1", 1), ("P", 3), ("RGB", 3), ("RGBA", 3)])
def test_bmp_matches_pillow(pil_mode, channels):
    from tensorcodec.decoders import decode_image

    pixels = np.random.default_rng(7).integers(0, 256, (5, 7, 4), np.uint8)
    source = Image.fromarray(pixels, "RGBA")
    output = BytesIO()
    (source if pil_mode == "RGBA" else source.convert(pil_mode)).save(output, "BMP")
    data = output.getvalue()
    # Pillow writes RGBA as 32-bit BI_RGB, whose fourth byte readers treat as padding.
    expected = np.asarray(Image.open(BytesIO(data)).convert("L" if channels == 1 else "RGB"))
    if channels == 1:
        expected = expected[..., None]
    unchanged = decode_image(data, mode="UNCHANGED")
    np.testing.assert_array_equal(unchanged, expected.transpose(2, 0, 1))
    rgb = expected if channels == 3 else np.repeat(expected, 3, axis=-1)
    np.testing.assert_array_equal(decode_image(data), rgb.transpose(2, 0, 1))


def test_bmp_alpha():
    from tensorcodec.decoders import decode_image, decode_png

    rgba = np.random.default_rng(8).integers(0, 256, (5, 7, 4), np.uint8)
    data = bitfields_bmp(rgba, 124, 0xFF000000)
    assert Image.open(BytesIO(data)).mode == "RGBA"
    np.testing.assert_array_equal(decode_image(data, mode="UNCHANGED"), rgba.transpose(2, 0, 1))
    np.testing.assert_array_equal(decode_image(data), rgba[..., :3].transpose(2, 0, 1))
    with pytest.raises(RuntimeError, match="expected png, got bmp"):
        decode_png(data)
    for header_size, mask in ((124, 0), (40, 0)):
        opaque = decode_image(bitfields_bmp(rgba, header_size, mask), mode="UNCHANGED")
        np.testing.assert_array_equal(opaque, rgba[..., :3].transpose(2, 0, 1))
