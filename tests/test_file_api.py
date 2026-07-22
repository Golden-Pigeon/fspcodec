import numpy as np
import pytest

import fspcodec


def arr(seed, h=32, w=45):
    rng = np.random.default_rng(seed)
    return (rng.random((h, w), dtype=np.float32) * 10 - 5).astype(np.float32)


def test_write_then_read(tmp_path):
    p = str(tmp_path / "a.fsp")
    a = arr(0)
    f = fspcodec.open(p, "w")
    f.write(a)
    f.close()
    assert f.closed
    f = fspcodec.open(p, "r")
    b = f.read()
    assert np.array_equal(a, b)
    assert f.read() is None  # EOF
    f.close()


def test_context_manager_and_props(tmp_path):
    p = str(tmp_path / "a.fsp")
    with fspcodec.open(p, "w") as f:
        assert (f.name, f.mode, f.closed) == (p, "w", False)
        f.write(arr(1))
    assert f.closed
    assert "closed" in repr(f)


def test_multi_array_stream_and_iteration(tmp_path):
    p = str(tmp_path / "s.fsp")
    arrays = [arr(2, 8, 8), arr(3, 16, 33), arr(4, 5, 7)]
    with fspcodec.open(p, "w") as f:
        for a in arrays:
            f.write(a)
    with fspcodec.open(p) as f:
        got = list(f)
    assert len(got) == len(arrays)
    for a, b in zip(arrays, got):
        assert np.array_equal(a, b)


def test_append_mode(tmp_path):
    p = str(tmp_path / "s.fsp")
    with fspcodec.open(p, "w") as f:
        f.write(arr(5))
    with fspcodec.open(p, "a") as f:
        f.write(arr(6))
    with fspcodec.open(p) as f:
        assert len(list(f)) == 2


def test_mode_errors(tmp_path):
    p = str(tmp_path / "a.fsp")
    with fspcodec.open(p, "w") as f:
        f.write(arr(7))
        with pytest.raises(ValueError):
            f.read()
    with fspcodec.open(p, "r") as f:
        with pytest.raises(ValueError):
            f.write(arr(8))
    with pytest.raises(ValueError):
        fspcodec.open(p, "x")


def test_closed_file_raises(tmp_path):
    p = str(tmp_path / "a.fsp")
    f = fspcodec.open(p, "w")
    f.close()
    with pytest.raises(ValueError):
        f.write(arr(9))
    f.close()  # idempotent


def test_interop_with_module_functions(tmp_path):
    p = str(tmp_path / "a.fsp")
    a = arr(10)
    fspcodec.write(p, a)  # module-level single-array writer
    with fspcodec.open(p) as f:
        assert np.array_equal(f.read(), a)
    p2 = str(tmp_path / "b.fsp")
    with fspcodec.open(p2, "w") as f:
        f.write(a)
    assert np.array_equal(fspcodec.read(p2), a)  # module-level reader


def test_rb_wb_aliases(tmp_path):
    p = str(tmp_path / "a.fsp")
    with fspcodec.open(p, "wb") as f:
        f.write(arr(11))
    with fspcodec.open(p, "rb") as f:
        assert f.read() is not None


def test_truncated_stream_raises(tmp_path):
    p = str(tmp_path / "s.fsp")
    with fspcodec.open(p, "w") as f:
        f.write(arr(12))
        f.write(arr(13))
    data = open(p, "rb").read()
    open(p, "wb").write(data[: len(data) - 50])
    with fspcodec.open(p) as f:
        assert f.read() is not None  # first blob intact
        with pytest.raises(ValueError):
            f.read()  # second blob truncated


def test_per_write_level_override(tmp_path):
    p = str(tmp_path / "a.fsp")
    a = arr(14, 128, 128)
    with fspcodec.open(p, "w", level=1) as f:
        f.write(a)
        f.write(a, level=9)
    with fspcodec.open(p) as f:
        assert all(np.array_equal(x, a) for x in f)
