"""Benchmark fspcodec against raw .npy round-trips.

Usage:
    python benchmarks/bench.py            # synthetic depth-like data
    python benchmarks/bench.py file.npy   # a real float array
"""

import sys
import time

import numpy as np

import fspcodec


def bench(fn, n=50):
    fn()
    t = time.perf_counter()
    for _ in range(n):
        r = fn()
    return (time.perf_counter() - t) / n * 1000, r


def main() -> None:
    if len(sys.argv) > 1:
        a = np.load(sys.argv[1]).astype(np.float32)
    else:
        y, x = np.mgrid[0:768, 0:1024].astype(np.float32)
        rng = np.random.default_rng(0)
        a = (2 + np.sin(x / 50) + np.cos(y / 40) + rng.normal(0, 0.02, x.shape)).astype(
            np.float32
        )
    for level in (1, 3, 9):
        te, blob = bench(lambda: fspcodec.compress(a, level=level), 20)
        td, out = bench(lambda: fspcodec.decompress(blob))
        assert np.array_equal(out, a)
        print(
            f"level={level}: {a.nbytes / 1e6:.2f} MB -> {len(blob) / 1e6:.3f} MB "
            f"({a.nbytes / len(blob):.2f}x)  enc={te:.2f} ms  dec={td:.2f} ms"
        )


if __name__ == "__main__":
    main()
