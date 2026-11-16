//! fspcodec — lossless byte-planar compression for float32 2-D arrays.
//!
//! FSP1 format (little-endian), 32-byte header:
//! ```text
//! 0..4    magic "FSP1"
//! 4       version (2)
//! 5       zstd level used at encode time (informational)
//! 6..8    reserved (0)
//! 8..12   height (u32)
//! 12..16  width  (u32)
//! 16..32  compressed sizes of planes b3,b2,b1,b0 (u32 each);
//!         size == H*W marks a raw (stored) plane
//! ```
//! followed by the four plane blobs back to back and an 8-byte XXH3-64
//! checksum of the uncompressed plane bytes (b3 || b2 || b1 || b0, in their
//! filtered on-disk form) — endianness-independent and covering every plane
//! including the raw one.
//!
//! Each float32 is split into its 4 bytes (b3 = sign+exponent high byte,
//! b0 = mantissa low byte). Planes b3..b1 are horizontally delta-filtered and
//! zstd-compressed; b0 is high-entropy mantissa noise and is stored raw, which
//! keeps both encode and decode fast. The transform is bit-exact, so NaN/Inf
//! payloads survive round-trips unchanged.
//!
//! The plane transforms have SSSE3 SIMD kernels with scalar fallback
//! (auto-detected; set FSPCODEC_NO_SIMD=1 to force scalar).
//!
//! Sparse records use the same magic and a version-3 header. They store
//! delta-coded u32 pixel indices and original u32 float bits in two optional
//! zstd blocks. The checksum covers the header and uncompressed blocks.

use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::fs::{File, OpenOptions};
use std::io::{Read as IoRead, Write as IoWrite};
use std::path::PathBuf;
use std::sync::OnceLock;
use xxhash_rust::xxh3::{xxh3_64, Xxh3};

const HDR: usize = 32;
const MAGIC: &[u8; 4] = b"FSP1";
const DENSE_VERSION: u8 = 2;
const SPARSE_VERSION: u8 = 3;
/// Upper bound on H*W accepted by encode and decode. Decoding allocates
/// roughly 7 * H * W bytes based on header fields alone, so this caps what a
/// hostile 32-byte header can make us allocate.
const MAX_PIXELS: usize = 1 << 31;

#[derive(Clone, Copy)]
enum Storage {
    Auto,
    Dense,
    Sparse,
}

impl Storage {
    fn parse(value: &str) -> PyResult<Self> {
        match value {
            "auto" => Ok(Self::Auto),
            "dense" => Ok(Self::Dense),
            "sparse" => Ok(Self::Sparse),
            _ => Err(verr("storage must be 'auto', 'dense' or 'sparse'")),
        }
    }
}

fn verr(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}

fn simd_enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| {
        if std::env::var_os("FSPCODEC_NO_SIMD").is_some() {
            return false;
        }
        #[cfg(target_arch = "x86_64")]
        {
            return std::arch::is_x86_feature_detected!("ssse3");
        }
        #[allow(unreachable_code)]
        false
    })
}

/// Scalar split+filter of one row starting at `start` with running previous
/// byte values `a3/a2/a1` (the previous pixel's original bytes).
#[allow(clippy::too_many_arguments)] // internal hot-path helper; grouping would cost clarity
fn split_row_scalar_from(
    row: &[u32],
    r3: &mut [u8],
    r2: &mut [u8],
    r1: &mut [u8],
    r0: &mut [u8],
    start: usize,
    mut a3: u8,
    mut a2: u8,
    mut a1: u8,
) {
    for i in start..row.len() {
        let v = row[i];
        let (b3, b2, b1) = ((v >> 24) as u8, (v >> 16) as u8, (v >> 8) as u8);
        r3[i] = b3.wrapping_sub(a3);
        r2[i] = b2.wrapping_sub(a2);
        r1[i] = b1.wrapping_sub(a1);
        r0[i] = v as u8;
        (a3, a2, a1) = (b3, b2, b1);
    }
}

/// Scalar unfilter+merge of one row starting at `start` with running sums.
#[allow(clippy::too_many_arguments)] // internal hot-path helper; grouping would cost clarity
fn unfilter_merge_row_scalar_from(
    r3: &[u8],
    r2: &[u8],
    r1: &[u8],
    r0: &[u8],
    drow: &mut [u32],
    start: usize,
    mut a3: u8,
    mut a2: u8,
    mut a1: u8,
) {
    for i in start..drow.len() {
        a3 = a3.wrapping_add(r3[i]);
        a2 = a2.wrapping_add(r2[i]);
        a1 = a1.wrapping_add(r1[i]);
        drow[i] = ((a3 as u32) << 24) | ((a2 as u32) << 16) | ((a1 as u32) << 8) | r0[i] as u32;
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use std::arch::x86_64::*;

    /// SSSE3 split+filter: de-interleave 16 pixels per iteration with byte
    /// shuffles, delta via a one-byte-shifted vector (alignr with the previous
    /// vector); the sub-16 tail runs scalar, seeded from the source pixels.
    #[target_feature(enable = "ssse3")]
    pub unsafe fn split_row(
        row: &[u32],
        r3: &mut [u8],
        r2: &mut [u8],
        r1: &mut [u8],
        r0: &mut [u8],
    ) {
        let w = row.len();
        // gather bytes of 4 pixels into [b0 x4 | b1 x4 | b2 x4 | b3 x4]
        let group = _mm_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
        let mut p3_prev = _mm_setzero_si128();
        let mut p2_prev = _mm_setzero_si128();
        let mut p1_prev = _mm_setzero_si128();
        let mut i = 0usize;
        while i + 16 <= w {
            let ptr = row.as_ptr().add(i) as *const __m128i;
            let s0 = _mm_shuffle_epi8(_mm_loadu_si128(ptr), group);
            let s1 = _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(1)), group);
            let s2 = _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(2)), group);
            let s3 = _mm_shuffle_epi8(_mm_loadu_si128(ptr.add(3)), group);
            let t0 = _mm_unpacklo_epi32(s0, s1);
            let t1 = _mm_unpackhi_epi32(s0, s1);
            let t2 = _mm_unpacklo_epi32(s2, s3);
            let t3 = _mm_unpackhi_epi32(s2, s3);
            let p0 = _mm_unpacklo_epi64(t0, t2);
            let p1 = _mm_unpackhi_epi64(t0, t2);
            let p2 = _mm_unpacklo_epi64(t1, t3);
            let p3 = _mm_unpackhi_epi64(t1, t3);
            let d3 = _mm_sub_epi8(p3, _mm_alignr_epi8(p3, p3_prev, 15));
            let d2 = _mm_sub_epi8(p2, _mm_alignr_epi8(p2, p2_prev, 15));
            let d1 = _mm_sub_epi8(p1, _mm_alignr_epi8(p1, p1_prev, 15));
            _mm_storeu_si128(r3.as_mut_ptr().add(i) as *mut __m128i, d3);
            _mm_storeu_si128(r2.as_mut_ptr().add(i) as *mut __m128i, d2);
            _mm_storeu_si128(r1.as_mut_ptr().add(i) as *mut __m128i, d1);
            _mm_storeu_si128(r0.as_mut_ptr().add(i) as *mut __m128i, p0);
            p3_prev = p3;
            p2_prev = p2;
            p1_prev = p1;
            i += 16;
        }
        if i < w {
            let (a3, a2, a1) = if i == 0 {
                (0, 0, 0)
            } else {
                let v = row[i - 1];
                ((v >> 24) as u8, (v >> 16) as u8, (v >> 8) as u8)
            };
            super::split_row_scalar_from(row, r3, r2, r1, r0, i, a3, a2, a1);
        }
    }

    /// SSSE3 unfilter+merge: log-step in-vector prefix sums plus a broadcast
    /// carry per 16 pixels, then byte interleave back into u32 pixels; the
    /// sub-16 tail runs scalar, seeded from the already-decoded pixels.
    #[target_feature(enable = "ssse3")]
    pub unsafe fn unfilter_merge_row(r3: &[u8], r2: &[u8], r1: &[u8], r0: &[u8], drow: &mut [u32]) {
        let w = drow.len();
        let b15 = _mm_set1_epi8(15);
        let mut c3 = _mm_setzero_si128();
        let mut c2 = _mm_setzero_si128();
        let mut c1 = _mm_setzero_si128();
        let mut i = 0usize;
        while i + 16 <= w {
            let mut v3 = _mm_loadu_si128(r3.as_ptr().add(i) as *const __m128i);
            let mut v2 = _mm_loadu_si128(r2.as_ptr().add(i) as *const __m128i);
            let mut v1 = _mm_loadu_si128(r1.as_ptr().add(i) as *const __m128i);
            let v0 = _mm_loadu_si128(r0.as_ptr().add(i) as *const __m128i);
            v3 = _mm_add_epi8(v3, _mm_slli_si128(v3, 1));
            v3 = _mm_add_epi8(v3, _mm_slli_si128(v3, 2));
            v3 = _mm_add_epi8(v3, _mm_slli_si128(v3, 4));
            v3 = _mm_add_epi8(v3, _mm_slli_si128(v3, 8));
            v3 = _mm_add_epi8(v3, c3);
            c3 = _mm_shuffle_epi8(v3, b15);
            v2 = _mm_add_epi8(v2, _mm_slli_si128(v2, 1));
            v2 = _mm_add_epi8(v2, _mm_slli_si128(v2, 2));
            v2 = _mm_add_epi8(v2, _mm_slli_si128(v2, 4));
            v2 = _mm_add_epi8(v2, _mm_slli_si128(v2, 8));
            v2 = _mm_add_epi8(v2, c2);
            c2 = _mm_shuffle_epi8(v2, b15);
            v1 = _mm_add_epi8(v1, _mm_slli_si128(v1, 1));
            v1 = _mm_add_epi8(v1, _mm_slli_si128(v1, 2));
            v1 = _mm_add_epi8(v1, _mm_slli_si128(v1, 4));
            v1 = _mm_add_epi8(v1, _mm_slli_si128(v1, 8));
            v1 = _mm_add_epi8(v1, c1);
            c1 = _mm_shuffle_epi8(v1, b15);
            let ab_lo = _mm_unpacklo_epi8(v0, v1);
            let ab_hi = _mm_unpackhi_epi8(v0, v1);
            let cd_lo = _mm_unpacklo_epi8(v2, v3);
            let cd_hi = _mm_unpackhi_epi8(v2, v3);
            let o = drow.as_mut_ptr().add(i) as *mut __m128i;
            _mm_storeu_si128(o, _mm_unpacklo_epi16(ab_lo, cd_lo));
            _mm_storeu_si128(o.add(1), _mm_unpackhi_epi16(ab_lo, cd_lo));
            _mm_storeu_si128(o.add(2), _mm_unpacklo_epi16(ab_hi, cd_hi));
            _mm_storeu_si128(o.add(3), _mm_unpackhi_epi16(ab_hi, cd_hi));
            i += 16;
        }
        if i < w {
            let (a3, a2, a1) = if i == 0 {
                (0, 0, 0)
            } else {
                let v = drow[i - 1];
                ((v >> 24) as u8, (v >> 16) as u8, (v >> 8) as u8)
            };
            super::unfilter_merge_row_scalar_from(r3, r2, r1, r0, drow, i, a3, a2, a1);
        }
    }
}

fn split_filter(src: &[u32], h: usize, w: usize, work: &mut [u8]) {
    let n = h * w;
    let (p3, rest) = work.split_at_mut(n);
    let (p2, rest) = rest.split_at_mut(n);
    let (p1, p0) = rest.split_at_mut(n);
    let use_simd = simd_enabled();
    for r in 0..h {
        let o = r * w;
        let row = &src[o..o + w];
        let (r3, r2, r1, r0) = (
            &mut p3[o..o + w],
            &mut p2[o..o + w],
            &mut p1[o..o + w],
            &mut p0[o..o + w],
        );
        #[cfg(target_arch = "x86_64")]
        if use_simd {
            unsafe { x86::split_row(row, r3, r2, r1, r0) };
            continue;
        }
        let _ = use_simd;
        split_row_scalar_from(row, r3, r2, r1, r0, 0, 0, 0, 0);
    }
}

fn encode_dense(src: &[f32], h: usize, w: usize, level: i32) -> Result<Vec<u8>, String> {
    let n = h * w;
    if n == 0 {
        return Err("array must be non-empty".into());
    }
    if h > u32::MAX as usize || w > u32::MAX as usize || n > MAX_PIXELS {
        return Err(format!("array dimensions {h}x{w} exceed format limits"));
    }
    let src_bits = unsafe { std::slice::from_raw_parts(src.as_ptr() as *const u32, n) };
    let mut work = vec![0u8; 4 * n];
    split_filter(src_bits, h, w, &mut work);

    // Worst case every plane is stored raw, plus the checksum trailer.
    let mut out = vec![0u8; HDR + 4 * n + 8];
    out[..4].copy_from_slice(MAGIC);
    out[4] = DENSE_VERSION;
    out[5] = level.clamp(0, 255) as u8;
    out[8..12].copy_from_slice(&(h as u32).to_le_bytes());
    out[12..16].copy_from_slice(&(w as u32).to_le_bytes());
    let mut off = HDR;
    for i in 0..4 {
        let plane = &work[i * n..(i + 1) * n];
        let cs = if i == 3 {
            // b0 is mantissa noise: storing raw beats compressing it.
            out[off..off + n].copy_from_slice(plane);
            n
        } else {
            match zstd::bulk::compress_to_buffer(plane, &mut out[off..off + n], level) {
                Ok(cs) if cs < n => cs,
                // Doesn't fit in n bytes => incompressible => store raw.
                _ => {
                    out[off..off + n].copy_from_slice(plane);
                    n
                }
            }
        };
        out[16 + 4 * i..20 + 4 * i].copy_from_slice(&(cs as u32).to_le_bytes());
        off += cs;
    }
    out[off..off + 8].copy_from_slice(&xxh3_64(&work).to_le_bytes());
    off += 8;
    out.truncate(off);
    Ok(out)
}

fn encode_block(raw: &[u8], level: i32) -> (Vec<u8>, bool) {
    if raw.is_empty() {
        return (Vec::new(), false);
    }
    let mut compressed = vec![0u8; raw.len()];
    match zstd::bulk::compress_to_buffer(raw, &mut compressed[..], level) {
        Ok(size) if size < raw.len() => {
            compressed.truncate(size);
            (compressed, true)
        }
        _ => (raw.to_vec(), false),
    }
}

fn encode_sparse(src: &[f32], h: usize, w: usize, level: i32) -> Result<Vec<u8>, String> {
    let mut indices = Vec::new();
    let mut values = Vec::new();
    let mut previous = 0u32;
    for (index, value) in src.iter().enumerate() {
        let bits = value.to_bits();
        // Only positive zero is implicit. Keep -0, infinities, subnormals and
        // every NaN payload exactly, without floating-point arithmetic.
        if bits != 0 {
            let index = index as u32;
            indices.extend_from_slice(&(index - previous).to_le_bytes());
            values.extend_from_slice(&bits.to_le_bytes());
            previous = index;
        }
    }
    let count = indices.len() / 4;
    let (index_blob, index_zstd) = encode_block(&indices, level);
    let (value_blob, value_zstd) = encode_block(&values, level);
    let index_size = u32::try_from(index_blob.len()).map_err(|_| "sparse index block too large")?;
    let value_size = u32::try_from(value_blob.len()).map_err(|_| "sparse value block too large")?;
    let mut header = [0u8; HDR];
    header[..4].copy_from_slice(MAGIC);
    header[4] = SPARSE_VERSION;
    header[5] = level.clamp(0, 255) as u8;
    header[6] = u8::from(index_zstd) | (u8::from(value_zstd) << 1);
    header[8..12].copy_from_slice(&(h as u32).to_le_bytes());
    header[12..16].copy_from_slice(&(w as u32).to_le_bytes());
    header[16..20].copy_from_slice(&(count as u32).to_le_bytes());
    header[20..24].copy_from_slice(&index_size.to_le_bytes());
    header[24..28].copy_from_slice(&value_size.to_le_bytes());
    let mut hasher = Xxh3::new();
    hasher.update(&header);
    hasher.update(&indices);
    hasher.update(&values);
    let mut out = Vec::with_capacity(HDR + index_blob.len() + value_blob.len() + 8);
    out.extend_from_slice(&header);
    out.extend_from_slice(&index_blob);
    out.extend_from_slice(&value_blob);
    out.extend_from_slice(&hasher.digest().to_le_bytes());
    Ok(out)
}

fn encode(
    src: &[f32],
    h: usize,
    w: usize,
    level: i32,
    storage: Storage,
) -> Result<Vec<u8>, String> {
    let n = h
        .checked_mul(w)
        .filter(|&n| n > 0 && n <= MAX_PIXELS)
        .ok_or("invalid dimensions")?;
    if h > u32::MAX as usize || w > u32::MAX as usize || src.len() != n {
        return Err("array dimensions exceed format limits".into());
    }
    match storage {
        Storage::Dense => encode_dense(src, h, w, level),
        Storage::Sparse => encode_sparse(src, h, w, level),
        Storage::Auto => {
            // Dense v2 always stores n raw low bytes. For at most n/8 explicit
            // values, even uncompressed sparse indices+bits cannot exceed that
            // mandatory plane. Dense data exits this scan early; do not encode
            // both alternatives or scan an entire dense image unnecessarily.
            let limit = n / 8;
            let mut count = 0usize;
            for value in src {
                if value.to_bits() != 0 {
                    count += 1;
                    if count > limit {
                        return encode_dense(src, h, w, level);
                    }
                }
            }
            encode_sparse(src, h, w, level)
        }
    }
}

struct RecordLayout {
    h: usize,
    w: usize,
    n: usize,
    version: u8,
    flags: u8,
    count: usize,
    sizes: [usize; 4],
    payload: usize,
}

fn record_layout(header: &[u8]) -> Result<RecordLayout, String> {
    if header.len() < HDR || &header[..4] != MAGIC {
        return Err("not an FSP1 blob".into());
    }
    let version = header[4];
    if version != DENSE_VERSION && version != SPARSE_VERSION {
        return Err(format!("unsupported FSP version {version}"));
    }
    let h = u32::from_le_bytes(header[8..12].try_into().unwrap()) as usize;
    let w = u32::from_le_bytes(header[12..16].try_into().unwrap()) as usize;
    let n = h
        .checked_mul(w)
        .filter(|&n| n > 0 && n <= MAX_PIXELS)
        .ok_or("invalid dimensions")?;

    let mut sizes = [0usize; 4];
    let mut count = 0usize;
    let flags = header[6];
    if version == DENSE_VERSION {
        for (i, size) in sizes.iter_mut().enumerate() {
            let cs =
                u32::from_le_bytes(header[16 + 4 * i..20 + 4 * i].try_into().unwrap()) as usize;
            if cs == 0 || cs > n {
                return Err("invalid dense plane size".into());
            }
            *size = cs;
        }
        // The v2 format specifies an uncompressed b0 plane. Reject malformed
        // fourth compressed planes before allocating its three-plane scratch.
        if sizes[3] != n {
            return Err("dense low-byte plane must be raw".into());
        }
    } else {
        if flags & !3 != 0 || header[7] != 0 || header[28..32] != [0; 4] {
            return Err("unsupported sparse flags or reserved fields".into());
        }
        count = u32::from_le_bytes(header[16..20].try_into().unwrap()) as usize;
        if count > n {
            return Err("sparse count exceeds image size".into());
        }
        let raw_size = count.checked_mul(4).ok_or("sparse size overflow")?;
        for (i, size) in sizes[..2].iter_mut().enumerate() {
            let cs =
                u32::from_le_bytes(header[20 + 4 * i..24 + 4 * i].try_into().unwrap()) as usize;
            let compressed = flags & (1 << i) != 0;
            if (compressed && (cs == 0 || cs >= raw_size)) || (!compressed && cs != raw_size) {
                return Err("invalid sparse block size".into());
            }
            *size = cs;
        }
    }
    let payload = sizes
        .iter()
        .try_fold(8usize, |sum, &size| sum.checked_add(size))
        .ok_or("record size overflow")?;
    Ok(RecordLayout {
        h,
        w,
        n,
        version,
        flags,
        count,
        sizes,
        payload,
    })
}

fn decode_sparse(buf: &[u8], layout: &RecordLayout) -> Result<(Vec<f32>, usize, usize), String> {
    let raw_size = layout.count.checked_mul(4).ok_or("sparse size overflow")?;
    let mut blocks = Vec::with_capacity(2);
    let mut off = HDR;
    for (i, &size) in layout.sizes[..2].iter().enumerate() {
        let input = &buf[off..off + size];
        let raw = if layout.flags & (1 << i) != 0 {
            // Allocate only the declared, bounded output length; never allow
            // a compressed frame to choose an unbounded decompression size.
            let mut raw = vec![0u8; raw_size];
            let decoded = zstd::bulk::decompress_to_buffer(input, &mut raw[..])
                .map_err(|e| format!("zstd: {e}"))?;
            if decoded != raw_size {
                return Err("sparse block size mismatch".into());
            }
            raw
        } else {
            input.to_vec()
        };
        blocks.push(raw);
        off += size;
    }
    let mut hasher = Xxh3::new();
    hasher.update(&buf[..HDR]);
    for block in &blocks {
        hasher.update(block);
    }
    let checksum = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
    if hasher.digest() != checksum {
        return Err("checksum mismatch (corrupt data)".into());
    }
    // Validate coordinates before allocating the restored image.
    let mut previous = 0usize;
    for (i, delta) in blocks[0].chunks_exact(4).enumerate() {
        let delta = u32::from_le_bytes(delta.try_into().unwrap()) as usize;
        if i > 0 && delta == 0 {
            return Err("sparse indices must be strictly increasing".into());
        }
        previous = previous
            .checked_add(delta)
            .filter(|&index| index < layout.n)
            .ok_or("sparse index outside image")?;
    }
    let mut out = vec![0f32; layout.n];
    let mut index = 0usize;
    for (delta, bits) in blocks[0].chunks_exact(4).zip(blocks[1].chunks_exact(4)) {
        index += u32::from_le_bytes(delta.try_into().unwrap()) as usize;
        out[index] = f32::from_bits(u32::from_le_bytes(bits.try_into().unwrap()));
    }
    Ok((out, layout.h, layout.w))
}

fn decode(buf: &[u8]) -> Result<(Vec<f32>, usize, usize), String> {
    let layout = record_layout(buf)?;
    let total = HDR
        .checked_add(layout.payload)
        .ok_or("record size overflow")?;
    if buf.len() < total {
        return Err("truncated or corrupt blob".into());
    }
    if layout.version == SPARSE_VERSION {
        return decode_sparse(buf, &layout);
    }
    let RecordLayout { h, w, n, sizes, .. } = layout;

    let mut work = vec![0u8; 3 * n];
    let mut tmp = &mut work[..];
    let mut planes: [&[u8]; 4] = [&[]; 4];
    let mut off = HDR;
    for (i, &cs) in sizes.iter().enumerate() {
        if cs == n {
            planes[i] = &buf[off..off + n];
        } else {
            let (dst, rest) = tmp.split_at_mut(n);
            let ds = zstd::bulk::decompress_to_buffer(&buf[off..off + cs], dst)
                .map_err(|e| format!("zstd: {e}"))?;
            if ds != n {
                return Err("plane size mismatch".into());
            }
            planes[i] = dst;
            tmp = rest;
        }
        off += cs;
    }

    if off + 8 > buf.len() {
        return Err("truncated or corrupt blob".into());
    }
    let expected = u64::from_le_bytes(buf[off..off + 8].try_into().unwrap());
    let mut hasher = Xxh3::new();
    for plane in &planes {
        hasher.update(plane);
    }
    if hasher.digest() != expected {
        return Err("checksum mismatch (corrupt data)".into());
    }

    let mut out = vec![0u32; n];
    let (p3, p2, p1, p0) = (planes[0], planes[1], planes[2], planes[3]);
    let use_simd = simd_enabled();
    for r in 0..h {
        let o = r * w;
        let (r3, r2, r1, r0) = (&p3[o..o + w], &p2[o..o + w], &p1[o..o + w], &p0[o..o + w]);
        let drow = &mut out[o..o + w];
        #[cfg(target_arch = "x86_64")]
        if use_simd {
            unsafe { x86::unfilter_merge_row(r3, r2, r1, r0, drow) };
            continue;
        }
        let _ = use_simd;
        unfilter_merge_row_scalar_from(r3, r2, r1, r0, drow, 0, 0, 0, 0);
    }
    let mut out = std::mem::ManuallyDrop::new(out);
    let out_f32 =
        unsafe { Vec::from_raw_parts(out.as_mut_ptr() as *mut f32, out.len(), out.capacity()) };
    Ok((out_f32, h, w))
}

/// Compress a float32 2-D array, automatically selecting dense or sparse storage.
#[pyfunction]
#[pyo3(signature = (array, level=3, *, storage="auto"))]
fn compress<'py>(
    py: Python<'py>,
    array: PyReadonlyArray2<'py, f32>,
    level: i32,
    storage: &str,
) -> PyResult<Bound<'py, PyBytes>> {
    let storage = Storage::parse(storage)?;
    let view = array.as_array();
    let std_layout = view.as_standard_layout();
    let (h, w) = std_layout.dim();
    let src = std_layout
        .as_slice()
        .ok_or_else(|| verr("failed to obtain contiguous view"))?;
    let blob = py
        .allow_threads(|| encode(src, h, w, level, storage))
        .map_err(verr)?;
    Ok(PyBytes::new(py, &blob))
}

/// Decompress a dense or sparse FSP1 blob back into a float32 2-D array.
#[pyfunction]
fn decompress<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (v, h, w) = py.allow_threads(|| decode(data)).map_err(verr)?;
    let arr = Array2::from_shape_vec((h, w), v).map_err(|e| verr(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

/// Compress `array` and write it to `path`.
#[pyfunction]
#[pyo3(signature = (path, array, level=3, *, storage="auto"))]
fn write<'py>(
    py: Python<'py>,
    path: PathBuf,
    array: PyReadonlyArray2<'py, f32>,
    level: i32,
    storage: &str,
) -> PyResult<()> {
    let storage = Storage::parse(storage)?;
    let view = array.as_array();
    let std_layout = view.as_standard_layout();
    let (h, w) = std_layout.dim();
    let src = std_layout
        .as_slice()
        .ok_or_else(|| verr("failed to obtain contiguous view"))?;
    py.allow_threads(|| {
        let blob = encode(src, h, w, level, storage)?;
        std::fs::write(&path, blob).map_err(|e| e.to_string())
    })
    .map_err(verr)
}

/// Read a dense or sparse FSP1 file into the original float32 2-D array.
#[pyfunction]
fn read<'py>(py: Python<'py>, path: PathBuf) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let (v, h, w) = py
        .allow_threads(|| {
            let data = std::fs::read(&path).map_err(|e| e.to_string())?;
            decode(&data)
        })
        .map_err(verr)?;
    let arr = Array2::from_shape_vec((h, w), v).map_err(|e| verr(e.to_string()))?;
    Ok(arr.into_pyarray(py))
}

#[derive(Clone, Copy, PartialEq)]
enum FileMode {
    Read,
    Write,
    Append,
}

fn read_exact_or_eof(f: &mut File, buf: &mut [u8]) -> Result<bool, String> {
    let mut got = 0usize;
    while got < buf.len() {
        let k = f.read(&mut buf[got..]).map_err(|e| e.to_string())?;
        if k == 0 {
            if got == 0 {
                return Ok(false);
            }
            return Err("truncated FSP stream".into());
        }
        got += k;
    }
    Ok(true)
}

/// Read one self-delimiting FSP1 blob from the stream; None at clean EOF.
fn read_next_blob(f: &mut File) -> Result<Option<Vec<u8>>, String> {
    let mut hdr = [0u8; HDR];
    if !read_exact_or_eof(f, &mut hdr)? {
        return Ok(None);
    }
    let layout = record_layout(&hdr)?;
    let total = HDR
        .checked_add(layout.payload)
        .ok_or("record size overflow")?;
    // Check the remaining file length before allocating a header-sized claim.
    let metadata = f.metadata().map_err(|e| e.to_string())?;
    if metadata.is_file() {
        let position = std::io::Seek::stream_position(f).map_err(|e| e.to_string())?;
        let remaining = metadata.len().saturating_sub(position);
        if (layout.payload as u64) > remaining {
            return Err("truncated FSP stream".into());
        }
    }
    let mut blob = vec![0u8; total];
    blob[..HDR].copy_from_slice(&hdr);
    if !read_exact_or_eof(f, &mut blob[HDR..])? {
        return Err("truncated FSP stream".into());
    }
    Ok(Some(blob))
}

/// A file object for FSP1 streams, created by [`open`].
///
/// A file holds one or more concatenated FSP1 blobs. `write` appends one
/// array per call; `read` returns the next array, or `None` at end of file.
/// The object is iterable and works as a context manager.
#[pyclass]
struct FspFile {
    file: Option<File>,
    mode: FileMode,
    path: PathBuf,
    level: i32,
    storage: Storage,
}

impl FspFile {
    fn read_impl<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f32>>>> {
        if self.mode != FileMode::Read {
            return Err(verr("file not open for reading"));
        }
        let f = self
            .file
            .as_mut()
            .ok_or_else(|| verr("I/O operation on closed file"))?;
        let res = py
            .allow_threads(|| -> Result<Option<(Vec<f32>, usize, usize)>, String> {
                match read_next_blob(f)? {
                    None => Ok(None),
                    Some(blob) => decode(&blob).map(Some),
                }
            })
            .map_err(verr)?;
        match res {
            None => Ok(None),
            Some((v, h, w)) => {
                let arr = Array2::from_shape_vec((h, w), v).map_err(|e| verr(e.to_string()))?;
                Ok(Some(arr.into_pyarray(py)))
            }
        }
    }
}

#[pymethods]
impl FspFile {
    /// Append one float32 2-D array to the file.
    #[pyo3(signature = (array, level=None, *, storage=None))]
    fn write<'py>(
        &mut self,
        py: Python<'py>,
        array: PyReadonlyArray2<'py, f32>,
        level: Option<i32>,
        storage: Option<&str>,
    ) -> PyResult<()> {
        if self.mode == FileMode::Read {
            return Err(verr("file not open for writing"));
        }
        let level = level.unwrap_or(self.level);
        let storage = storage
            .map(Storage::parse)
            .transpose()?
            .unwrap_or(self.storage);
        let view = array.as_array();
        let std_layout = view.as_standard_layout();
        let (h, w) = std_layout.dim();
        let src = std_layout
            .as_slice()
            .ok_or_else(|| verr("failed to obtain contiguous view"))?;
        let f = self
            .file
            .as_mut()
            .ok_or_else(|| verr("I/O operation on closed file"))?;
        py.allow_threads(|| {
            let blob = encode(src, h, w, level, storage)?;
            f.write_all(&blob).map_err(|e| e.to_string())
        })
        .map_err(verr)
    }

    /// Read the next array from the file, or None at end of file.
    fn read<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f32>>>> {
        self.read_impl(py)
    }

    /// Close the file. Further reads/writes raise ValueError.
    fn close(&mut self) {
        self.file = None;
    }

    #[getter]
    fn closed(&self) -> bool {
        self.file.is_none()
    }

    #[getter]
    fn name(&self) -> String {
        self.path.display().to_string()
    }

    #[getter]
    fn mode(&self) -> &'static str {
        match self.mode {
            FileMode::Read => "r",
            FileMode::Write => "w",
            FileMode::Append => "a",
        }
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &mut self,
        _exc_type: &Bound<'_, PyAny>,
        _exc_value: &Bound<'_, PyAny>,
        _traceback: &Bound<'_, PyAny>,
    ) -> bool {
        self.file = None;
        false
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__<'py>(&mut self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray2<f32>>>> {
        self.read_impl(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "<fspcodec.FspFile name='{}' mode='{}'{}>",
            self.path.display(),
            match self.mode {
                FileMode::Read => "r",
                FileMode::Write => "w",
                FileMode::Append => "a",
            },
            if self.file.is_none() { " closed" } else { "" }
        )
    }
}

/// Open an FSP file and return a file object.
///
/// mode: "r" (read), "w" (create/truncate), "a" (append). A trailing "b" is
/// accepted and ignored. `level` sets the default zstd level for writes.
#[pyfunction]
#[pyo3(signature = (path, mode="r", level=3, *, storage="auto"))]
fn open(path: PathBuf, mode: &str, level: i32, storage: &str) -> PyResult<FspFile> {
    let storage = Storage::parse(storage)?;
    let m = match mode.trim_end_matches('b') {
        "r" => FileMode::Read,
        "w" => FileMode::Write,
        "a" => FileMode::Append,
        _ => return Err(verr(format!("invalid mode '{mode}' (use 'r', 'w' or 'a')"))),
    };
    let file = match m {
        FileMode::Read => File::open(&path),
        FileMode::Write => File::create(&path),
        FileMode::Append => OpenOptions::new().append(true).create(true).open(&path),
    }
    .map_err(|e| verr(format!("{}: {e}", path.display())))?;
    Ok(FspFile {
        file: Some(file),
        mode: m,
        path,
        level,
        storage,
    })
}

#[pymodule]
fn fspcodec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(compress, m)?)?;
    m.add_function(wrap_pyfunction!(decompress, m)?)?;
    m.add_function(wrap_pyfunction!(write, m)?)?;
    m.add_function(wrap_pyfunction!(read, m)?)?;
    m.add_function(wrap_pyfunction!(open, m)?)?;
    m.add_class::<FspFile>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
