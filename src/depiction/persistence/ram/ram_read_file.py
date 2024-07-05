from __future__ import annotations
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path

from depiction.persistence.ram.ram_reader import RamReader
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from depiction.persistence import ImzmlModeEnum
    from collections.abc import Generator
    from numpy.typing import NDArray


class RamReadFile:
    def __init__(
        self,
        mz_arr_list: list[NDArray[float]] | NDArray[float],
        int_arr_list: list[NDArray[float]] | NDArray[float],
        coordinates: NDArray[int],
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
    def reader(self) -> Generator[RamReader, None, None]:
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
    def coordinates(self) -> NDArray[int]:
        return self._coordinates

    @property
    def coordinates_2d(self) -> NDArray[int]:
        return self._coordinates[:, :2]
