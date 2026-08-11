from __future__ import annotations
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path

from depiction_io.ram.ram_reader import RamReader
from depiction_io.types import GenericReadFile
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from depiction_io import ImzmlModeEnum
    from depiction_io.pixel_size import PixelSize
    from collections.abc import Generator
    from numpy.typing import NDArray


class RamReadFile(GenericReadFile):
    def __init__(
        self,
        mz_arr_list: list[NDArray[np.float64]] | NDArray[np.float64],
        int_arr_list: list[NDArray[np.float64]] | NDArray[np.float64],
        coordinates: NDArray[np.int64],
    ) -> None:
        self._mz_arr_list = mz_arr_list
        self._int_arr_list = int_arr_list
        self._coordinates = coordinates

    @property
    def imzml_file(self) -> Path:
        print("Warning: imzml_file is not available for RamReadFile")
        return Path("/dev/null")

    @property
    def ibd_file(self) -> Path:
        print("Warning: ibd_file is not available for RamReadFile")
        return Path("/dev/null")

    @contextmanager
    def reader(self) -> Generator[RamReader]:
        reader = self.get_reader()
        try:
            yield reader
        finally:
            reader.close()

    def get_reader(self) -> RamReader:
        return RamReader(mz_arr_list=self._mz_arr_list, int_arr_list=self._int_arr_list, coordinates=self._coordinates)

    @property
    def n_spectra(self) -> int:
        return len(self._mz_arr_list)

    @cached_property
    def imzml_mode(self) -> ImzmlModeEnum:
        with self.reader() as reader:
            return reader.imzml_mode

    @property
    def coordinates(self) -> NDArray[np.int64]:
        return self._coordinates

    @property
    def coordinates_2d(self) -> NDArray[np.int64]:
        return self._coordinates[:, :2]

    @property
    def compact_metadata(self) -> dict[str, int | str | list[float]]:
        """Compact representation of general metadata about the spectra held in memory.

        Deliberately without the `imzml_file` / `ibd_file` keys `ImzyReadFile` provides: there is
        no file, and this dict exists to be compared across acquisitions, where a placeholder path
        is worse than an absent key.
        """
        return {
            "n_spectra": self.n_spectra,
            "imzml_mode": self.imzml_mode.name,
            "coordinate_extent": list(self.coordinates.max(0) - self.coordinates.min(0) + 1),
        }

    @property
    def is_checksum_valid(self) -> bool | None:
        """Always `None` -- there is no `.ibd` to checksum, which the protocol spells `None`."""
        return None

    @property
    def pixel_size(self) -> PixelSize | None:
        """Always `None` -- spectra held in memory carry no declared raster.

        Present so a tool can forward `read_file.pixel_size` to its output without caring which
        backend it was handed.
        """
        return None

    def summary(self, checksums: bool = True) -> str:
        """A summary of the spectra held in memory.

        `checksums` is accepted and ignored, for signature compatibility with the file-backed
        implementations: there is no `.ibd` here, so there is nothing to verify either way.
        """
        return f"in-memory spectra\nimzML mode: {self.imzml_mode.name}\nn_spectra: {self.n_spectra}\n"
