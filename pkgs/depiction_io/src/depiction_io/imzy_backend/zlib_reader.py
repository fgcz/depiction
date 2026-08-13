"""imzy's imzML reader, taught to undo zlib compression.

imzy reads a binary array as ``array_length * itemsize`` raw bytes at the array's offset.
For a zlib-compressed array both halves of that are wrong: the bytes on disk are the
*compressed* ones, and how many of them there are (``IMS:1000104``) is a number imzy never
parses -- ``byte_offsets`` carries only ``IMS:1000103``, the element count. The result is
not an error but noise, which is why this exists.

Everything else imzy does -- parsing the XML, the offset table, the coordinate list, the
`.icache` sidecar -- is unaffected by compression and is left alone. Only the three places
that turn bytes into floats are overridden, and `_ENCODED_READ_SITES` pins that list so an
imzy upgrade that adds a fourth fails loudly here rather than silently returning noise
again.

# upstream: no compression support in imzy. See docs/modules/depiction_io/imzy_backend.md, gap (1).
"""

from __future__ import annotations

import zlib
from typing import TYPE_CHECKING, Any

import numpy as np
from imzy import IMZMLReader

if TYPE_CHECKING:
    import typing as ty
    from collections.abc import Iterable, Iterator

    from koyo.typing import PathLike
    from numpy.typing import DTypeLike, NDArray

#: Every method of `IMZMLReader` that reads encoded bytes out of the `.ibd`. All three are
#: overridden below; the assertion that follows is what catches upstream growing a fourth.
_ENCODED_READ_SITES = ("_read_spectrum", "_read_spectra", "_estimate_centroid_mass_range")

_missing = [name for name in _ENCODED_READ_SITES if not hasattr(IMZMLReader, name)]
if _missing:
    raise ImportError(
        f"imzy's IMZMLReader no longer defines {', '.join(_missing)}. The zlib backend overrides those methods to"
        f" decompress; if they were renamed or removed, compressed files would silently be read as raw floats again."
    )


class ZlibIMZMLReader(IMZMLReader):
    """An `imzy.IMZMLReader` that decompresses zlib-compressed binary arrays.

    Args:
        path: the .imzML file to read.
        encoded_lengths: `IMS:1000102` offset -> `IMS:1000104` encoded length, as produced
            by `depiction_io.imzy_backend.imzml_scan.scan_imzml`.
    """

    def __init__(self, path: PathLike, encoded_lengths: dict[int, int], **kwargs: Any) -> None:
        # Set before `super().__init__`, which parses the file and may reach a read site.
        self._encoded_lengths = encoded_lengths
        super().__init__(path, **kwargs)

    def _read_block(self, f_ptr: ty.BinaryIO, offset: int, array_length: int, dtype: DTypeLike) -> NDArray[Any]:
        """Reads one compressed binary array and returns it decoded.

        Args:
            f_ptr: an open handle on the .ibd.
            offset: the array's `IMS:1000102` offset.
            array_length: the array's `IMS:1000103` element count, used as a check.
            dtype: the precision imzy parsed for this array.

        Raises:
            ValueError: the offset was not in the scan, or the block did not decode to the
                length the imzML declares -- both of which mean the scan and the .ibd
                describe different files.
        """
        offset = int(offset)
        try:
            encoded_length = self._encoded_lengths[offset]
        except KeyError:
            raise ValueError(
                f"No encoded length for the binary array at offset {offset} of {self.path}. The imzML scan and the"
                f" file it was taken from have diverged."
            ) from None

        f_ptr.seek(offset)
        array = np.frombuffer(zlib.decompress(f_ptr.read(encoded_length)), dtype=dtype)
        if array.size != array_length:
            raise ValueError(
                f"The binary array at offset {offset} of {self.path} decoded to {array.size} values, but the imzML"
                f" declares {array_length}."
            )
        return array

    def _read_pair(self, f_ptr: ty.BinaryIO, index: int) -> tuple[NDArray[Any], NDArray[Any]]:
        """Reads the m/z and intensity arrays of one spectrum from an open .ibd."""
        mz_o, mz_l, int_o, int_l = self.byte_offsets[index]
        return (
            self._read_block(f_ptr, mz_o, mz_l, self.mz_precision),
            self._read_block(f_ptr, int_o, int_l, self.int_precision),
        )

    def _read_spectrum(self, index: int) -> tuple[NDArray[Any], NDArray[Any]]:
        with self.ibd_path.open("rb") as f_ptr:
            return self._read_pair(f_ptr, index)

    def _read_spectra(self, indices: Iterable[int] | None = None) -> Iterator[tuple[NDArray[Any], NDArray[Any]]]:
        if indices is None:
            indices = self.pixels
        with self.ibd_path.open("rb") as f_ptr:
            for index in indices:
                yield self._read_pair(f_ptr, index)

    def _estimate_centroid_mass_range(self) -> tuple[float, float]:
        # Unreachable through this package's adapter, which never touches `mz_min`, `mz_max`
        # or `mz_x` -- but it is a read site, and leaving one of those un-decompressed is
        # how the noise gets back in.
        mz_min, mz_max = np.inf, -np.inf
        with self.ibd_path.open("rb") as f_ptr:
            for index in self.pixels:
                mz_o, mz_l, _, _ = self.byte_offsets[index]
                mz_arr = self._read_block(f_ptr, mz_o, mz_l, self.mz_precision)
                if mz_arr.size == 0:
                    continue
                mz_min = min(mz_min, float(np.min(mz_arr)))
                mz_max = max(mz_max, float(np.max(mz_arr)))
        if not np.isfinite(mz_min) or not np.isfinite(mz_max):
            raise ValueError("Cannot determine m/z range from empty centroid spectra.")
        return float(mz_min), float(mz_max)
