import pathlib

import numpy as np
import pytest

import fspcodec


def arr(h=16, w=16):
    return np.random.default_rng(0).random((h, w), dtype=np.float32)


def test_pathlib_paths_accepted(tmp_path):
    a = arr()
    p = tmp_path / "a.fsp"  # pathlib.Path, not str
    fspcodec.write(p, a)
    assert np.array_equal(fspcodec.read(p), a)
    with fspcodec.open(p) as f:
        assert np.array_equal(f.read(), a)
    with fspcodec.open(tmp_path / "b.fsp", "w") as f:
        f.write(a)
    assert isinstance(f.name, str)


def _forged_header(h, w, plane_sizes):
    hdr = b"FSP1" + bytes([2, 3, 0, 0])
    hdr += h.to_bytes(4, "little") + w.to_bytes(4, "little")
    for cs in plane_sizes:
        hdr += cs.to_bytes(4, "little")
    return hdr


def test_huge_dims_header_rejected_fast():
    # A hostile 32-byte header claiming a ~10^19-pixel array must be rejected
    # before any allocation, not OOM.
    blob = _forged_header(2**32 - 1, 2**32 - 1, [8, 8, 8, 8]) + b"x" * 32
    with pytest.raises(ValueError):
        fspcodec.decompress(blob)
    blob = _forged_header(1 << 20, 1 << 20, [8, 8, 8, 8]) + b"x" * 32  # n = 2**40
    with pytest.raises(ValueError):
        fspcodec.decompress(blob)


def test_plane_size_beyond_buffer_rejected():
    blob = _forged_header(4, 4, [16, 16, 16, 16])  # sizes claim raw planes, no payload
    with pytest.raises(ValueError):
        fspcodec.decompress(blob)


def test_hostile_stream_rejected(tmp_path):
    p = tmp_path / "s.fsp"
    p.write_bytes(_forged_header(1 << 20, 1 << 20, [8, 8, 8, 8]) + b"x" * 32)
    with fspcodec.open(p) as f:
        with pytest.raises(ValueError):
            f.read()
