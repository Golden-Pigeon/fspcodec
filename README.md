# fspcodec

Lossless byte-planar compression for float32 2-D arrays, built for depth maps
and similar spatially smooth data. Rust + [PyO3], compression by [zstd].

Float32 depth maps are hard to compress: the mantissa's low bits are noise.
fspcodec splits each value into its 4 bytes ("byte planes"), delta-filters the
structured planes along rows, compresses them with zstd, and stores the noise
plane raw. The result beats 16-bit-PNG-based lossless schemes on size *and*
speed, in a single file:

| scheme (768x1024 depth map, float64 source 6.29 MB) | files | size | encode | decode |
|---|---|---|---|---|
| two 16-bit PNGs (hi/lo of float32)   | 2 | 1.56 MB | 78 ms | 25 ms |
| three PNGs (16 + 8 + 8)              | 3 | 1.32 MB | 68 ms | 24 ms |
| **fspcodec (zstd level 3)**            | **1** | **1.20 MB** | **7 ms** | **2.5 ms** |

Measured on 150 [Hypersim] depth maps (mean): 5.2x smaller than raw float64,
decode ~2.5 ms per 768x1024 map, bit-exact round-trip incl. NaN/Inf/subnormals.
Plane transforms use SSSE3 SIMD kernels when available (scalar fallback,
`FSPCODEC_NO_SIMD=1` to force it); SIMD and scalar paths produce byte-identical
files.
Your numbers will vary with content smoothness.

## Install

Requires a Rust toolchain and Python >= 3.9.

```bash
pip install maturin
maturin develop --release        # or: pip install .
```

## Usage

```python
import numpy as np
import fspcodec

depth = np.load("depth.npy").astype(np.float32)   # 2-D float32

blob = fspcodec.compress(depth, level=3)            # -> bytes (FSP1 container)
restored = fspcodec.decompress(blob)                # -> float32 ndarray, bit-exact

fspcodec.write("depth.fsp", depth, level=9)         # level 9: ~5% smaller, slower encode
restored = fspcodec.read("depth.fsp")
```

### File objects

`fspcodec.open` returns a file object supporting sequential read/write of one or
more arrays per file (blobs are self-delimiting), context management and
iteration:

```python
with fspcodec.open("frames.fsp", "w", level=3) as f:   # "w" truncate, "a" append
    f.write(frame0)
    f.write(frame1, level=9)                          # per-write override

with fspcodec.open("frames.fsp") as f:                  # mode "r" is the default
    first = f.read()                                  # next array, None at EOF
    for frame in f:                                   # or just iterate the rest
        ...
```

All heavy work runs with the GIL released, so multi-worker PyTorch dataloaders
scale naturally.

## FSP1 format

Little-endian, 32-byte header followed by four plane blobs:

| offset | field |
|---|---|
| 0  | magic `"FSP1"` |
| 4  | version (2) |
| 5  | zstd level used (informational) |
| 6  | reserved |
| 8  | height (u32) |
| 12 | width (u32) |
| 16 | compressed sizes of planes b3, b2, b1, b0 (4 x u32); size == H*W means the plane is stored raw |

The four plane blobs follow the header back to back, then an 8-byte XXH3-64
checksum of the uncompressed plane bytes (endianness-independent, covers the
raw plane too). Decoding verifies it and fails on any corruption.

Planes: byte k of each float32 (b3 = sign + exponent high byte ... b0 =
mantissa low byte). b3-b1 are horizontally delta-filtered (wrapping u8
subtraction) before zstd; b0 is always stored raw.

## Scope & limits

- float32, 2-D, C-order (non-contiguous inputs are copied transparently)
- bit-exact: NaN payloads, infinities, -0.0 and subnormals survive round-trips
- paths accept `str` and `os.PathLike`
- arrays are capped at 2**31 elements (encode and decode)
- integrity: an XXH3-64 checksum over all plane data is verified on decode
- trust model: decoding allocates roughly `7 * H * W` bytes based on header
  fields alone (capped at 2**31 elements) — still validate sizes before
  decoding untrusted blobs
- not an image format: no color, no metadata; pair it with a sidecar if needed

## License

MIT
