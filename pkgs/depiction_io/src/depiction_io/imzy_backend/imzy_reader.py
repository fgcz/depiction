from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from depiction_io.imzml.imzml_mode_enum import ImzmlModeEnum
from depiction_io.types import GenericReader

if TYPE_CHECKING:
    from imzy import BaseReader
    from numpy.typing import NDArray


class ImzyReader(GenericReader):
    """Reads spectra through `imzy`, behind the `GenericReader` protocol.

    This holds no file handle and no offset table of its own: imzy opens the `.ibd` per
    read call and keeps the offsets on its own reader object.
    """

    def __init__(
        self, path: str | Path, declares_z: bool = True, encoded_lengths: dict[int, int] | None = None
    ) -> None:
        """
        Args:
            path: the .imzML (or Bruker .d) file to read.
            declares_z: whether `coordinates` should include the z column. imzy always
                reports one; see `imzml_scan` for why it must not be passed on blindly.
            encoded_lengths: the offset -> encoded length map of a zlib-compressed file, or
                None when it is uncompressed. Its presence is what selects the decompressing
                reader.
        """
        self._path = Path(path)
        self._declares_z = declares_z
        self._encoded_lengths = encoded_lengths
        self._reader: BaseReader | None = None

    def __getstate__(self) -> dict[str, Any]:
        # imzy readers hold an offset table parsed from the file and are not picklable, so
        # the state is the path and the things that cannot be recovered from it cheaply.
        # `_get_reader_kwargs()` returns `{}` for imzML, so re-opening by path is lossless.
        # The encoded lengths travel rather than being re-derived: recovering them means
        # another walk of the XML, which is the expensive half of opening a large file.
        # upstream: imzy readers implement neither __getstate__ nor __setstate__; see
        # ROADMAP.md, Phase D gap (2).
        return {"path": self._path, "declares_z": self._declares_z, "encoded_lengths": self._encoded_lengths}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self._path = state["path"]
        self._declares_z = state["declares_z"]
        self._encoded_lengths = state["encoded_lengths"]
        self._reader = None

    @property
    def path(self) -> Path:
        """The path of the file being read."""
        return self._path

    @property
    def reader(self) -> BaseReader:
        """The underlying imzy reader, opened on first use."""
        if self._reader is None:
            self._discard_stale_icache()
            # imzy writes an `.icache` sidecar next to the input, so a read-only input tree
            # silently degrades to a full re-parse on every open.
            # upstream: no cache_dir argument; see ROADMAP.md, Phase D gap (3).
            if self._encoded_lengths is None:
                import imzy

                self._reader = imzy.get_reader(self._path)
            else:
                # `get_reader` dispatches on the file suffix and would hand back the plain
                # imzML reader, which cannot decompress.
                from depiction_io.imzy_backend.zlib_reader import ZlibIMZMLReader

                self._reader = ZlibIMZMLReader(self._path, encoded_lengths=self._encoded_lengths)
        return self._reader

    def _discard_stale_icache(self) -> None:
        """Removes imzy's offset cache when it predates the file it describes.

        imzy reloads `<stem>.icache` on sight -- no size, mtime or UUID check -- so a cache
        left over from an earlier file at the same path makes it report that file's spectrum
        count and coordinates while slicing the current `.ibd`. Overwriting through
        `ImzmlWriteFile` clears the cache already; this catches the paths that do not go
        through it, such as a file replaced by an external tool.

        Deleting a cache that was in fact valid costs a re-parse, so the comparison
        deliberately errs towards deleting. A cache that cannot be removed is left to raise:
        reading through a stale one is the outcome this exists to prevent.
        # upstream: the cache carries no provenance; see ROADMAP.md, Phase D gap (3).
        """
        icache = self._path.with_suffix(".icache")
        if not icache.exists():
            return
        if icache.stat().st_mtime <= self._path.stat().st_mtime:
            icache.unlink()

    def close(self) -> None:
        """Drops the underlying reader. imzy holds no persistent handle, so this frees only
        the parsed offset table."""
        self._reader = None

    @cached_property
    def imzml_mode(self) -> ImzmlModeEnum:
        """Returns the mode of the imzML file.

        imzy has no notion of continuous vs processed, so the mode is inferred from the offset
        table: all spectra sharing one m/z offset means continuous. This is what the parser
        this backend replaced did, which is why the swap did not change any tool's behaviour.
        Reading `IMS:1000030`/`IMS:1000031` instead would disagree on a single-spectrum file,
        which the corpus pins down deliberately.
        """
        byte_offsets = getattr(self.reader, "byte_offsets", None)
        if byte_offsets is None:
            # A vendor reader: there is no `.ibd` offset table to infer sharing from, and
            # no vendor format stores one m/z axis for the whole acquisition.
            return ImzmlModeEnum.PROCESSED
        return ImzmlModeEnum.CONTINUOUS if len(np.unique(byte_offsets[:, 0])) == 1 else ImzmlModeEnum.PROCESSED

    @property
    def n_spectra(self) -> int:
        """The number of spectra available in the file."""
        return self.reader.n_pixels

    @cached_property
    def coordinates(self) -> NDArray[np.int64]:
        """Returns the coordinates of the spectra, shape (n_spectra, n_dim)."""
        coordinates = np.asarray(self.reader.xyz_coordinates, dtype=np.int64)
        return coordinates if self._declares_z else coordinates[:, :2]

    def get_spectrum(self, i_spectrum: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Returns the m/z and intensity arrays of the i-th spectrum.

        Overridden because imzy reads both arrays and reopens the `.ibd` on every call, so
        the protocol's default -- one call per array -- would open the file twice and read
        each array twice. This is the hot path in nearly every tool.
        """
        return self.reader.get_spectrum(i_spectrum)

    def get_spectrum_mz(self, i_spectrum: int) -> NDArray[np.float64]:
        """Returns the m/z values of the i-th spectrum."""
        return self.reader.get_spectrum(i_spectrum)[0]

    def get_spectrum_int(self, i_spectrum: int) -> NDArray[np.float64]:
        """Returns the intensity values of the i-th spectrum."""
        return self.reader.get_spectrum(i_spectrum)[1]

    def get_spectrum_n_points(self, i_spectrum: int) -> int:
        """Returns the number of data points in the i-th spectrum, without reading it.

        This is `IMS:1000103`, the element count. The parser this backend replaced returned
        `IMS:1000104` instead -- the *encoded length* in bytes, four times larger for an
        uncompressed float32 array. That was a long-standing bug: its only caller is
        `depiction.tools.experimental.msi_hdf5`, and the test that would have caught it was
        skipped in its entirety. So this method now returns something different from what it
        used to, on purpose.
        """
        byte_offsets = getattr(self.reader, "byte_offsets", None)
        if byte_offsets is None:
            return len(self.get_spectrum_mz(i_spectrum))
        return int(byte_offsets[i_spectrum, 3])

    def get_spectra(
        self, i_spectra: list[int]
    ) -> tuple[NDArray[np.float64] | list[NDArray[np.float64]], NDArray[np.float64] | list[NDArray[np.float64]]]:
        """Returns the m/z and intensity arrays of the specified spectra.

        Routed onto imzy's batched read, which opens the `.ibd` once for the whole chunk
        rather than once per spectrum. That is the difference that shows up under
        `ReadSpectraParallel`: imzy seek/reads rather than mmapping, so the protocol's default
        implementation of this method would pay one `open()` per spectrum.
        """
        # `_read_spectra` is private; a public batched read is what imzy is missing here.
        pairs = list(self.reader._read_spectra(i_spectra))
        if self.imzml_mode == ImzmlModeEnum.CONTINUOUS:
            mz_arr_list = np.repeat(pairs[0][0][np.newaxis, :], len(i_spectra), axis=0)
            return mz_arr_list, np.stack([int_arr for _, int_arr in pairs], axis=0)
        return tuple(zip(*pairs))

    def __str__(self) -> str:
        return f"ImzyReader[{self._path}, n_spectra={self.n_spectra}]"
