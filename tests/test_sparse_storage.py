"""Automatic sparse FSP storage preserves the complete float32 bit pattern."""

import struct

import numpy as np
import pytest

import fspcodec


def assert_bits_equal(a, b):
    assert b.dtype == np.float32
    assert a.shape == b.shape
    np.testing.assert_array_equal(a.view(np.uint32), b.view(np.uint32))


def sparse_depth(h=768, w=1024, count=1000, seed=123):
    rng = np.random.default_rng(seed)
    a = np.zeros((h, w), dtype=np.float32)
    selected = rng.choice(a.size, count, replace=False)
    a.flat[selected] = rng.uniform(0.1, 150.0, count).astype(np.float32)
    return a


@pytest.mark.parametrize("shape", [(768, 1024), (375, 1242)])
@pytest.mark.parametrize("count", [500, 1000, 1500])
def test_realistic_sparse_storage_is_smaller_and_bit_exact(shape, count):
    a = sparse_depth(*shape, count=count)
    automatic = fspcodec.compress(a)
    forced_sparse = fspcodec.compress(a, storage="sparse")
    forced_dense = fspcodec.compress(a, storage="dense")
    assert automatic[:5] == b"FSP1\x03"
    assert automatic == forced_sparse
    assert forced_dense[:5] == b"FSP1\x02"
    assert len(automatic) < len(forced_dense) * 0.05
    assert_bits_equal(a, fspcodec.decompress(automatic))
    assert_bits_equal(a, fspcodec.decompress(forced_dense))


@pytest.mark.parametrize("shape", [(1, 1), (1, 17), (32, 32), (768, 1024)])
def test_all_positive_zero_uses_header_and_checksum_only(shape):
    a = np.zeros(shape, dtype=np.float32)
    blob = fspcodec.compress(a)
    assert blob[:5] == b"FSP1\x03"
    assert len(blob) == 40
    assert struct.unpack_from("<III", blob, 16) == (0, 0, 0)
    assert blob[6:8] == b"\x00\x00"
    assert_bits_equal(a, fspcodec.decompress(blob))


@pytest.mark.parametrize("storage", ["auto", "dense", "sparse"])
def test_special_float32_payloads_are_never_dropped_or_canonicalized(storage):
    bits = np.zeros((32, 64), dtype=np.uint32)
    values = np.array(
        [
            0x00000000,  # +0.0: implicit background
            0x80000000,  # -0.0: must remain explicit
            0x7F800000,  # positive infinity
            0xFF800000,  # negative infinity
            0x7FC12345,  # quiet NaN with a nontrivial payload
            0x7F800001,  # signaling NaN payload
            0xFFCABCDE,  # negative quiet NaN payload
            0x00000001,  # smallest positive subnormal
            0x80000001,  # smallest negative subnormal
            0x00800000,  # smallest positive normal
            0x7F7FFFFF,  # largest positive finite value
            0xFF7FFFFF,  # largest negative finite value
        ],
        dtype=np.uint32,
    )
    bits.flat[np.arange(len(values)) * 13] = values
    a = bits.view(np.float32)
    blob = fspcodec.compress(a, storage=storage)
    if storage != "dense":
        assert blob[4] == 3
        assert struct.unpack_from("<I", blob, 16)[0] == len(values) - 1
    assert_bits_equal(a, fspcodec.decompress(blob))


@pytest.mark.parametrize("storage", ["auto", "dense", "sparse"])
@pytest.mark.parametrize("layout", ["transpose", "slice", "reverse", "fortran"])
def test_noncontiguous_sparse_inputs(storage, layout):
    base = sparse_depth(96, 128, count=150)
    if layout == "transpose":
        a = base.T
    elif layout == "slice":
        a = base[::2, ::3]
    elif layout == "reverse":
        a = base[::-1, ::-1]
    else:
        a = np.asfortranarray(base)
    assert not a.flags.c_contiguous
    assert_bits_equal(a, fspcodec.decompress(fspcodec.compress(a, storage=storage)))


@pytest.mark.parametrize("shape", [(1, 1), (1, 2), (2, 1), (2, 3), (7, 5)])
@pytest.mark.parametrize("storage", ["auto", "dense", "sparse"])
def test_tiny_nonzero_arrays(shape, storage):
    a = np.arange(1, np.prod(shape) + 1, dtype=np.float32).reshape(shape)
    blob = fspcodec.compress(a, storage=storage)
    assert_bits_equal(a, fspcodec.decompress(blob))
    assert blob[4] == (3 if storage == "sparse" else 2)


@pytest.mark.parametrize("shape", [(0, 0), (0, 8), (8, 0)])
@pytest.mark.parametrize("storage", ["auto", "dense", "sparse"])
def test_empty_arrays_remain_rejected(shape, storage):
    with pytest.raises(ValueError):
        fspcodec.compress(np.zeros(shape, dtype=np.float32), storage=storage)


def test_dense_legacy_golden_bytes_are_unchanged():
    # Captured from the original version-2 writer before the sparse implementation.
    expected = bytes.fromhex(
        "4653503102030000010000000100000001000000010000000100000001000000"
        "3fc00000cac9ce5daef15ef0"
    )
    a = np.array([[1.5]], dtype=np.float32)
    assert fspcodec.compress(a, storage="dense") == expected
    assert fspcodec.compress(a) == expected
    assert_bits_equal(a, fspcodec.decompress(expected))


def test_default_dense_encoding_is_identical_to_explicit_dense():
    a = np.random.default_rng(41).random((80, 91), dtype=np.float32)
    automatic = fspcodec.compress(a)
    assert automatic[4] == 2
    assert automatic == fspcodec.compress(a, storage="dense")
    assert_bits_equal(a, fspcodec.decompress(automatic))


def test_auto_threshold_includes_exact_eight_bytes_per_entry():
    a = np.zeros((1, 64), dtype=np.float32)
    a.flat[:8] = np.arange(1, 9, dtype=np.float32)
    assert fspcodec.compress(a)[4] == 3
    a.flat[8] = 9
    assert fspcodec.compress(a)[4] == 2


def test_sparse_header_and_raw_small_blocks():
    a = np.zeros((3, 5), dtype=np.float32)
    a[1, 2] = 1.5
    blob = fspcodec.compress(a, storage="sparse")
    assert blob[:5] == b"FSP1\x03"
    assert blob[6:8] == b"\x00\x00"
    assert blob[28:32] == b"\x00" * 4
    assert struct.unpack_from("<IIIII", blob, 8) == (3, 5, 1, 4, 4)
    assert struct.unpack_from("<II", blob, 32) == (7, 0x3FC00000)
    assert len(blob) == 48
    assert_bits_equal(a, fspcodec.decompress(blob))


def test_sequential_positions_and_values_use_compressed_blocks():
    a = np.zeros((256, 256), dtype=np.float32)
    a.flat[:2000] = 3.25
    blob = fspcodec.compress(a, storage="sparse")
    assert blob[6] == 3  # index and value payloads both benefit from zstd
    assert_bits_equal(a, fspcodec.decompress(blob))


def test_sparse_module_file_io_and_storage_override(tmp_path):
    a = sparse_depth(64, 80, count=20)
    p = tmp_path / "sparse.fsp"
    fspcodec.write(p, a)
    assert p.read_bytes()[4] == 3
    assert_bits_equal(a, fspcodec.read(p))
    fspcodec.write(p, a, storage="dense")
    assert p.read_bytes()[4] == 2
    assert_bits_equal(a, fspcodec.read(p))


def test_stream_mixes_legacy_dense_and_sparse_records(tmp_path):
    p = tmp_path / "mixed.fsp"
    arrays = [
        sparse_depth(40, 50, count=10),
        np.arange(35, dtype=np.float32).reshape(5, 7),
        np.zeros((1, 1), dtype=np.float32),
        sparse_depth(32, 48, count=15),
    ]
    with fspcodec.open(p, "w", storage="sparse") as f:
        f.write(arrays[0])
        f.write(arrays[1], storage="dense")
        f.write(arrays[2], storage="auto")
    with fspcodec.open(p, "a", storage="dense") as f:
        f.write(arrays[3], storage="sparse")
    with fspcodec.open(p) as f:
        restored = list(f)
    assert len(restored) == len(arrays)
    for a, b in zip(arrays, restored):
        assert_bits_equal(a, b)
    assert_bits_equal(arrays[0], fspcodec.read(p))


@pytest.mark.parametrize("storage", ["", "invalid", "Sparse", "AUTO"])
def test_unknown_storage_raises_for_all_writers(tmp_path, storage):
    a = np.zeros((3, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="storage"):
        fspcodec.compress(a, storage=storage)
    with pytest.raises(ValueError, match="storage"):
        fspcodec.write(tmp_path / "bad.fsp", a, storage=storage)
    with pytest.raises(ValueError, match="storage"):
        fspcodec.open(tmp_path / "bad.fsp", "w", storage=storage)
    with fspcodec.open(tmp_path / "stream.fsp", "w") as f:
        with pytest.raises(ValueError, match="storage"):
            f.write(a, storage=storage)


@pytest.mark.parametrize("offset", [5, 8, 12, 32, 36, -1])
def test_sparse_header_payload_and_checksum_corruption_detected(offset):
    a = np.zeros((3, 5), dtype=np.float32)
    a[1, 2] = 1.5
    blob = bytearray(fspcodec.compress(a, storage="sparse"))
    blob[offset] ^= 1
    with pytest.raises(ValueError):
        fspcodec.decompress(bytes(blob))


@pytest.mark.parametrize("length", [0, 4, 31, 32, 35, 39, 40, 44, 47])
def test_sparse_truncation_is_rejected(length):
    a = np.zeros((3, 5), dtype=np.float32)
    a[1, 2] = 1.5
    blob = fspcodec.compress(a, storage="sparse")
    with pytest.raises(ValueError):
        fspcodec.decompress(blob[:length])


@pytest.mark.parametrize(
    "offset,replacement",
    [
        (6, b"\x04"),  # undefined flags
        (7, b"\x01"),  # reserved byte
        (28, b"\x01"),  # reserved word
        (8, struct.pack("<I", 0)),
        (12, struct.pack("<I", 0)),
        (16, struct.pack("<I", 16)),  # nnz exceeds 15 pixels
        (20, struct.pack("<I", 0xFFFFFFFF)),
        (24, struct.pack("<I", 0xFFFFFFFF)),
    ],
)
def test_sparse_invalid_header_rejected_before_allocating(tmp_path, offset, replacement):
    a = np.zeros((3, 5), dtype=np.float32)
    a[1, 2] = 1.5
    blob = bytearray(fspcodec.compress(a, storage="sparse"))
    blob[offset : offset + len(replacement)] = replacement
    with pytest.raises(ValueError):
        fspcodec.decompress(bytes(blob))
    p = tmp_path / "bad.fsp"
    p.write_bytes(blob)
    with fspcodec.open(p) as f:
        with pytest.raises(ValueError):
            f.read()


def test_truncated_sparse_record_in_mixed_stream(tmp_path):
    p = tmp_path / "mixed_truncated.fsp"
    first = np.arange(25, dtype=np.float32).reshape(5, 5)
    second = sparse_depth(64, 64, count=30)
    p.write_bytes(
        fspcodec.compress(first, storage="dense")
        + fspcodec.compress(second, storage="sparse")[:-1]
    )
    with fspcodec.open(p) as f:
        assert_bits_equal(first, f.read())
        with pytest.raises(ValueError):
            f.read()


def test_forced_sparse_preserves_random_float32_bit_patterns_at_full_density():
    rng = np.random.default_rng(912)
    bits = rng.integers(0, 2**32, size=(43, 57), dtype=np.uint32)
    a = bits.view(np.float32)
    blob = fspcodec.compress(a, storage="sparse")
    assert blob[4] == 3
    assert_bits_equal(a, fspcodec.decompress(blob))


def test_negative_zero_is_not_implicit_background():
    a = np.full((7, 13), np.uint32(0x80000000), dtype=np.uint32).view(np.float32)
    blob = fspcodec.compress(a, storage="sparse")
    assert struct.unpack_from("<I", blob, 16)[0] == a.size
    assert_bits_equal(a, fspcodec.decompress(blob))


@pytest.mark.parametrize("storage", ["auto", "dense", "sparse"])
def test_sparse_options_do_not_change_dtype_contract(storage):
    with pytest.raises(TypeError):
        fspcodec.compress(np.zeros((5, 7), dtype=np.float64), storage=storage)
    with pytest.raises(TypeError):
        fspcodec.compress(np.zeros((5, 7, 1), dtype=np.float32), storage=storage)


@pytest.mark.parametrize(
    "message_hex,checksum,error",
    [
        (
            "4653503103030000020000000200000001000000040000000400000000000000"
            "040000000000c03f",
            0x054FD985419C886A,
            "index|bounds|position",
        ),
        (
            "4653503103030000020000000200000002000000080000000800000000000000"
            "01000000000000000000803f00000040",
            0xC888533528FAF3F1,
            "index|duplicate|increasing|delta",
        ),
        (
            "4653503103030000020000000200000002000000080000000800000000000000"
            "03000000ffffffff0000803f00000040",
            0x7408C640FD71764C,
            "index|bounds|overflow|position",
        ),
    ],
    ids=["out_of_bounds", "duplicate_index", "index_accumulation_overflow"],
)
def test_valid_checksum_cannot_hide_invalid_sparse_indices(
    tmp_path, message_hex, checksum, error
):
    # XXH3 constants computed independently by the xxhash-rust crate over the
    # exact header + uncompressed index/value bytes. No optional Python package
    # or skipped test is needed, and rejection must concern index semantics.
    blob = bytes.fromhex(message_hex) + struct.pack("<Q", checksum)
    with pytest.raises(ValueError, match=error):
        fspcodec.decompress(blob)
    p = tmp_path / "invalid_index.fsp"
    p.write_bytes(blob)
    with fspcodec.open(p) as f:
        with pytest.raises(ValueError, match=error):
            f.read()
