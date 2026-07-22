"""fspcodec — lossless byte-planar compression for float32 2-D arrays.

Each float32 is split into its 4 bytes (byte planes); the structured planes
are delta-filtered and zstd-compressed, the noise plane is stored raw. The
transform is bit-exact: NaN payloads, infinities, ``-0.0`` and subnormals
survive round-trips unchanged. Heavy work runs with the GIL released.

The container format is FSP1 (see README for the byte-level layout).
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

def compress(array: npt.NDArray[np.float32], level: int = 3) -> bytes:
    """Compress a float32 2-D array into an FSP1 blob.

    Args:
        array: 2-D ``float32`` array. Non-contiguous input is copied
            transparently; other dtypes raise ``TypeError``.
        level: zstd compression level (1-22). Higher is smaller but slower;
            3 is a good speed/size default, 9 gains ~5% for slow encodes.

    Returns:
        The FSP1 blob. Round-trips bit-exactly through :func:`decompress`.

    Raises:
        ValueError: If the array is empty or exceeds 2**31 elements.
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
        An XXH3-64 checksum over all plane data is verified before returning.
        Decoding allocates roughly ``7 * H * W`` bytes based on header fields
        (capped at 2**31 elements), so still apply an external size check
        before decoding untrusted blobs.

    Raises:
        ValueError: If the blob is truncated, corrupt, fails the checksum,
            or is not FSP1.
    """

def write(path: _StrPath, array: npt.NDArray[np.float32], level: int = 3) -> None:
    """Compress ``array`` and write a single-array FSP1 file to ``path``.

    Equivalent to ``open(path, "w", level).write(array)`` for one array.

    Raises:
        ValueError: On encoding failure or I/O errors.
        TypeError: If the array is not 2-D float32.
    """

def read(path: _StrPath) -> npt.NDArray[np.float32]:
    """Read an FSP1 file and return the (first) decoded array.

    For multi-array stream files written via :func:`open`, only the first
    array is returned; use :func:`open` to iterate over all of them.

    Raises:
        ValueError: If the file is missing, truncated, or not FSP1.
    """

def open(
    path: _StrPath,
    mode: Literal["r", "w", "a", "rb", "wb", "ab"] = "r",
    level: int = 3,
) -> FspFile:
    """Open an FSP file and return a file object.

    Args:
        path: Filesystem path.
        mode: ``"r"`` read (default), ``"w"`` create/truncate, ``"a"``
            append. A trailing ``"b"`` is accepted and ignored.
        level: Default zstd level for writes on this file.

    Returns:
        An :class:`FspFile`. Use it as a context manager where possible.

    Raises:
        ValueError: For invalid modes or if the file cannot be opened.
    """

@final
class FspFile:
    """A file object for FSP1 streams, created by :func:`open`.

    A file holds one or more concatenated FSP1 records. ``write`` appends one
    array per call; ``read`` returns the next array, or ``None`` at end of
    file. The object is iterable and works as a context manager::

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

    def write(self, array: npt.NDArray[np.float32], level: int | None = None) -> None:
        """Append one float32 2-D array to the file.

        Args:
            array: 2-D ``float32`` array.
            level: Per-write zstd level override; defaults to the level
                given to :func:`open`.

        Raises:
            ValueError: If the file is closed or open for reading.
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
