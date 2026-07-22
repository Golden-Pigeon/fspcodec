import numpy as np
import pytest

import fspcodec


def smooth_depth(h=240, w=320, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    depth = 2.0 + np.sin(x / 40) + np.cos(y / 30) + rng.normal(0, 0.01, (h, w))
    return depth.astype(np.float32)


def test_roundtrip_smooth():
    a = smooth_depth()
    blob = fspcodec.compress(a)
    b = fspcodec.decompress(blob)
    assert b.dtype == np.float32 and b.shape == a.shape
    assert np.array_equal(a, b)


def test_roundtrip_noise_raw_fallback():
    rng = np.random.default_rng(1)
    a = rng.random((64, 64), dtype=np.float32)
    blob = fspcodec.compress(a)
    assert np.array_equal(fspcodec.decompress(blob), a)


def test_bit_exact_special_values():
    a = smooth_depth(32, 32)
    a[0, 0] = np.nan
    a[0, 1] = np.inf
    a[0, 2] = -np.inf
    a[0, 3] = -0.0
    a[0, 4] = np.float32(1e-42)  # subnormal
    b = fspcodec.decompress(fspcodec.compress(a))
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))


def test_non_contiguous_input():
    a = smooth_depth(64, 64)[::2, ::2]
    assert not a.flags.c_contiguous
    assert np.array_equal(fspcodec.decompress(fspcodec.compress(a)), a)


def test_levels_change_size_not_content():
    a = smooth_depth()
    b1, b9 = fspcodec.compress(a, level=1), fspcodec.compress(a, level=9)
    assert len(b9) <= len(b1)
    assert np.array_equal(fspcodec.decompress(b1), fspcodec.decompress(b9))


def test_header():
    a = smooth_depth(17, 23)
    blob = fspcodec.compress(a)
    assert blob[:4] == b"FSP1" and blob[4] == 2
    h = int.from_bytes(blob[8:12], "little")
    w = int.from_bytes(blob[12:16], "little")
    assert (h, w) == (17, 23)


def test_compresses_smooth_data():
    a = smooth_depth(480, 640)
    assert len(fspcodec.compress(a)) < a.nbytes * 0.75


def test_corrupt_blob_raises():
    with pytest.raises(ValueError):
        fspcodec.decompress(b"not an fsp blob at all")
    a = smooth_depth(16, 16)
    blob = bytearray(fspcodec.compress(a))
    blob[16:20] = (2**31).to_bytes(4, "little")  # absurd plane size
    with pytest.raises(ValueError):
        fspcodec.decompress(bytes(blob))


def test_truncated_blob_raises():
    blob = fspcodec.compress(smooth_depth(32, 32))
    with pytest.raises(ValueError):
        fspcodec.decompress(blob[: len(blob) // 2])


def test_wrong_dtype_raises():
    with pytest.raises(TypeError):
        fspcodec.compress(np.zeros((4, 4), dtype=np.float64))


def test_file_roundtrip(tmp_path):
    a = smooth_depth()
    p = str(tmp_path / "x.fsp")
    fspcodec.write(p, a)
    assert np.array_equal(fspcodec.read(p), a)


def test_read_missing_file_raises(tmp_path):
    with pytest.raises(ValueError):
        fspcodec.read(str(tmp_path / "missing.fsp"))


def test_bitflip_detected_by_checksum():
    a = smooth_depth(64, 64)
    blob = bytearray(fspcodec.compress(a))
    # flip one bit in the raw b0 plane (the last plane, just before the trailer)
    blob[-12] ^= 0x01
    with pytest.raises(ValueError, match="checksum"):
        fspcodec.decompress(bytes(blob))


def test_trailer_truncation_detected():
    blob = fspcodec.compress(smooth_depth(16, 16))
    with pytest.raises(ValueError):
        fspcodec.decompress(blob[:-4])
