"""fspcodec — lossless dense/sparse compression for float32 2-D arrays.

Dense storage uses delta-filtered byte planes and zstd; sparse storage keeps
pixel-index deltas and original value bits. The default selects sparse storage
when ``8 * nnz <= H * W``, counting nonzero float32 bit patterns. NaN payloads,
infinities, ``-0.0`` and subnormals survive bit-exactly. Heavy work runs with the
GIL released.

FSP1 readers identify dense version 2 and sparse version 3 automatically.
Writing with ``storage="dense"`` preserves compatibility with older readers;
upgrading the library does not rewrite existing files. See README for layouts.
"""

from os import PathLike
from types import TracebackType
from typing import Literal, TypeAlias, final

import numpy as np
import numpy.typing as npt

__all__ = [
    "FspFile",
    "__version__",
    "compress",
    "decompress",
    "open",
    "read",
    "write",
]

__version__: str

_StrPath: TypeAlias = str | PathLike[str]
_Storage: TypeAlias = Literal["auto", "dense", "sparse"]

def compress(
    array: npt.NDArray[np.float32],
    level: int = 3,
    *,
    storage: _Storage = "auto",
) -> bytes:
    """Compress a float32 2-D array into an FSP1 blob.

    Args:
        array: 2-D ``float32`` array. Non-contiguous input is copied
            transparently; other dtypes raise ``TypeError``.
        level: zstd compression level (1-22). Higher is smaller but slower;
            3 is a good speed/size default, 9 gains ~5% for slow encodes.
        storage: ``"auto"`` uses sparse storage when ``8 * nnz <= H * W``;
            ``nnz`` counts nonzero float32 bit patterns, including ``-0.0``.
            ``"dense"`` forces the unchanged version-2 byte-plane encoding
            without a sparse scan. ``"sparse"`` forces version-3 index/value
            encoding. This argument is keyword-only.

    Returns:
        The FSP1 blob. Round-trips bit-exactly through :func:`decompress`.

    Raises:
        ValueError: If the array is empty, exceeds 2**31 elements, or the
            storage mode is invalid.
        TypeError: If the array is not 2-D float32.
    """

def decompress(data: bytes) -> npt.NDArray[np.float32]:
    """Decompress an FSP1 blob back into a float32 2-D array.

    Args:
        data: A blob produced by :func:`compress` (or one FSP1 record of a
            stream written via :func:`open`). Trailing bytes are ignored.

    Returns:
        The decoded 2-D ``float32`` array, bit-exact to the original.

    Note:
        Dense version 2 and sparse version 3 are identified from the header.
        The XXH3-64 checksum covers dense plane data, or the sparse header and
        decoded index/value segments. Decoding allocates a dense result plus
        format-dependent temporary buffers from header fields (capped at
        2**31 elements); apply size checks before decoding untrusted blobs.

    Raises:
        ValueError: If the blob is truncated, corrupt, fails the checksum,
            or is not FSP1.
    """

def write(
    path: _StrPath,
    array: npt.NDArray[np.float32],
    level: int = 3,
    *,
    storage: _Storage = "auto",
) -> None:
    """Compress ``array`` and write a single-array FSP1 file to ``path``.

    Equivalent to ``open(path, "w", level, storage=storage).write(array)``
    for one array. ``storage`` is keyword-only and accepts ``"auto"``,
    ``"dense"`` or ``"sparse"`` with the same meaning as :func:`compress`.

    Raises:
        ValueError: On encoding failure, invalid storage mode, or I/O errors.
        TypeError: If the array is not 2-D float32.
    """

def read(path: _StrPath) -> npt.NDArray[np.float32]:
    """Read an FSP1 file and return the (first) decoded array.

    Dense version 2 and sparse version 3 are identified automatically. For
    multi-array stream files written via :func:`open`, only the first array
    is returned; use :func:`open` to iterate over all of them.

    Raises:
        ValueError: If the file is missing, truncated, or not FSP1.
    """

def open(
    path: _StrPath,
    mode: Literal["r", "w", "a", "rb", "wb", "ab"] = "r",
    level: int = 3,
    *,
    storage: _Storage = "auto",
) -> FspFile:
    """Open an FSP file and return a file object.

    Args:
        path: Filesystem path.
        mode: ``"r"`` read (default), ``"w"`` create/truncate, ``"a"``
            append. A trailing ``"b"`` is accepted and ignored.
        level: Default zstd level for writes on this file.
        storage: Keyword-only default storage for writes: ``"auto"``,
            ``"dense"`` or ``"sparse"``. Reading identifies each record's
            encoding automatically, regardless of this write default.

    Returns:
        An :class:`FspFile`. Use it as a context manager where possible.

    Raises:
        ValueError: For invalid modes/storage or if the file cannot be opened.
    """

@final
class FspFile:
    """A file object for FSP1 streams, created by :func:`open`.

    A file holds one or more concatenated FSP1 records, which may mix dense
    version 2 and sparse version 3. ``write`` appends one array per call;
    ``read`` returns the next array, or ``None`` at end of file. The object is
    iterable and works as a context manager::

        with fspcodec.open("frames.fsp", "w") as f:
            f.write(frame0)
            f.write(frame1, level=9)

        with fspcodec.open("frames.fsp") as f:
            for frame in f:
                ...

    Not constructible directly — use :func:`open`.
    """

    @property
    def closed(self) -> bool:
        """True after :meth:`close` (or leaving a ``with`` block)."""

    @property
    def name(self) -> str:
        """The path passed to :func:`open`."""

    @property
    def mode(self) -> Literal["r", "w", "a"]:
        """The normalized open mode."""

    def write(
        self,
        array: npt.NDArray[np.float32],
        level: int | None = None,
        *,
        storage: _Storage | None = None,
    ) -> None:
        """Append one float32 2-D array to the file.

        Args:
            array: 2-D ``float32`` array.
            level: Per-write zstd level override; defaults to the level
                given to :func:`open`.
            storage: Keyword-only per-write override (``"auto"``,
                ``"dense"`` or ``"sparse"``). ``None`` inherits the mode
                given to :func:`open`.

        Raises:
            ValueError: If the file is closed, open for reading, or the
                storage mode is invalid.
            TypeError: If the array is not 2-D float32.
        """

    def read(self) -> npt.NDArray[np.float32] | None:
        """Read the next array from the file, or ``None`` at end of file.

        Raises:
            ValueError: If the file is closed, open for writing, or the
                stream is truncated/corrupt.
        """

    def close(self) -> None:
        """Close the file (idempotent). Further reads/writes raise ``ValueError``."""

    def __enter__(self) -> FspFile: ...
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> bool: ...
    def __iter__(self) -> FspFile: ...
    def __next__(self) -> npt.NDArray[np.float32]: ...
