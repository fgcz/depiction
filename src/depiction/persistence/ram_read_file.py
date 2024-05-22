from contextlib import contextmanager
from functools import cached_property
from collections.abc import Generator
from pathlib import Path

from depiction.persistence import ImzmlModeEnum
from depiction.persistence.ram_reader import RamReader


class RamReadFile:
    def __init__(self, mz_arr_list, int_arr_list, coordinates) -> None:
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
    def coordinates(self):
        return self._coordinates

    @property
    def coordinates_2d(self):
        return self._coordinates[:, :2]
