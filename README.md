# fspcodec

Lossless compression for float32 2-D arrays, built for dense and sparse depth
maps. Rust + [PyO3], compression by [zstd]. Sparse storage is selected
automatically when appropriate; the read API restores the same dense ndarray.

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

# The default storage="auto" selects sparse or dense storage from the data.
fspcodec.write("sparse.fsp", sparse_depth)
restored = fspcodec.read("sparse.fsp")               # same API and ndarray shape

# Explicit controls; storage is keyword-only.
blob = fspcodec.compress(depth, storage="dense")     # version 2 for older readers
blob = fspcodec.compress(sparse_depth, storage="sparse")
```

### Automatic sparse storage

`compress(array, level=3, *, storage="auto")` and
`write(path, array, level=3, *, storage="auto")` accept three storage modes:

- `"auto"` (default): use sparse storage if `8 * nnz <= H * W`, otherwise
  retain the existing dense encoding. `nnz` counts float32 values whose **bit
  pattern is not zero**, so `-0.0`, NaNs, infinities and subnormals are stored
  explicitly and round-trip bit-exactly.
- `"dense"`: always use the existing byte-plane encoding, without a sparse
  eligibility scan. This also produces files readable by older version-2
  readers.
- `"sparse"`: explicitly store pixel indices and float32 value bits, even if
  the array exceeds the automatic density threshold.

The automatic threshold bounds the uncompressed sparse payload (one u32 index
and one u32 value per stored pixel) by `H * W` bytes: the size of the low-byte
plane that dense storage always writes raw. It avoids encoding both formats
and comparing them. Once the scan exceeds the threshold, it stops and uses
dense storage; for a fully nonzero array, this happens after reading roughly
the first eighth of the values. Sparse index and value segments each choose
the smaller of raw and zstd storage. A nonempty array containing only positive
zeros needs just the 32-byte header and 8-byte checksum.

Readers identify the encoding from each record's header. Upgrading the
library **does not convert existing files**: the storage choice applies when
writing new files. New readers support the existing dense version 2 and sparse
version 3; older readers cannot read version 3. Use `storage="dense"` when
producing files for an older environment.

### File objects

`fspcodec.open` returns a file object supporting sequential read/write of one or
more arrays per file (blobs are self-delimiting), context management and
iteration:

```python
with fspcodec.open("frames.fsp", "w", level=3, storage="auto") as f:
    # "w" truncates; "a" appends. Each record has its own encoding.
    f.write(frame0)
    f.write(frame1, level=9)                          # per-write override
    f.write(frame2, storage="dense")                  # per-write storage override

with fspcodec.open("frames.fsp") as f:                  # mode "r" is the default
    first = f.read()                                  # next array, None at EOF
    for frame in f:                                   # or just iterate the rest
        ...
```

`open(path, mode="r", level=3, *, storage="auto")` sets the default write
storage. `FspFile.write(array, level=None, *, storage=None)` inherits both
defaults when they are `None`. A stream may mix dense and sparse records;
reading requires no storage argument.

All heavy work runs with the GIL released, so multi-worker PyTorch dataloaders
scale naturally.

## FSP1 format

The magic stays `"FSP1"`. The version byte selects the dense or sparse record
layout; both have a 32-byte header and an 8-byte checksum trailer, and records
remain self-delimiting for sequential streams.

### Dense records (version 2, unchanged)

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

### Sparse records (version 3)

All multibyte fields are little-endian.

| offset | field |
|---|---|
| 0 | magic `"FSP1"` (4 bytes) |
| 4 | version (3) |
| 5 | zstd level used (informational) |
| 6 | flags: bit 0 = zstd index segment; bit 1 = zstd value segment |
| 7 | reserved, zero |
| 8 | height (u32) |
| 12 | width (u32) |
| 16 | number of stored pixels, `nnz` (u32) |
| 20 | stored index-segment length in bytes (u32) |
| 24 | stored value-segment length in bytes (u32) |
| 28 | reserved (4 bytes), zero |

The index segment and then the value segment follow the header. Indices are
increasing row-major flat pixel indices, represented as u32 deltas (the first
index is relative to zero). Values are the corresponding original float32 bit
patterns as u32. Each decoded segment contains `4 * nnz` bytes; each is stored
raw or zstd-compressed, whichever is smaller, with the flags identifying the
choice. Decoding starts from a positive-zero dense array and restores the
stored values at their indices.

The trailer is an XXH3-64 checksum over the header followed by the decoded
index and value segments. The decoder verifies the checksum as well as segment
sizes and index validity. With no stored pixels, both segments are empty and
the complete record is 40 bytes.

## Sparse depth measurements

On 64 real sparse-depth cache inputs (32 Hypersim and 32 VKITTI, randomly
selected before inspecting outputs, 500/1000/1500 input points), version 0.2
gave the following means. KB uses 1000 bytes. Read timings include restoring
the full native FP32 array, with warm filesystem cache; they do not measure
cold-disk throughput. All reconstructed float bits matched the supplied inputs.

| Inputs | Dense FSP size | Automatic FSP size | Dense read | Automatic read |
|---|---:|---:|---:|---:|
| Hypersim, 768x1024 | 800.78 KB | 5.48 KB | 0.886 ms | 0.134 ms |
| VKITTI, 375x1242 | 480.05 KB | 5.46 KB | 0.560 ms | 0.088 ms |

This reduced sparse-depth file bytes by 99.32% and 98.86%, respectively.
The benchmark compares the two encodings in the same new library; it is not
a whole-training throughput measurement. Reproduce on synthetic inputs with
`python benchmarks/bench_sparse.py --output /tmp/fsp-sparse.json`, or supply
`--paths-json` with existing FSP paths to benchmark actual observations.

## Scope & limits

- float32, 2-D, C-order (non-contiguous inputs are copied transparently)
- bit-exact: NaN payloads, infinities, -0.0 and subnormals survive round-trips
- paths accept `str` and `os.PathLike`
- arrays are capped at 2**31 elements (encode and decode)
- integrity: an XXH3-64 checksum is verified on decode (dense plane data, or
  sparse header and decoded segments)
- trust model: decoding allocates a dense `4 * H * W`-byte result plus
  format-dependent temporary buffers based on header fields (capped at
  2**31 elements) — still validate sizes before decoding untrusted blobs
- not an image format: no color, no metadata; pair it with a sidecar if needed

## License

MIT
