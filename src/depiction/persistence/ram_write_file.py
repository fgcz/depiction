from __future__ import annotations
from tqdm import tqdm
from contextlib import contextmanager
from typing import TYPE_CHECKING


from depiction.persistence import ImzmlModeEnum, RamReadFile

if TYPE_CHECKING:
    from collections.abc import Sequence
    import numpy as np


class RamWriteFile:
    def __init__(self, imzml_mode: ImzmlModeEnum) -> None:
        self._mz_arr_list = []
        self._int_arr_list = []
        self._coordinates_list = []
        self._imzml_mode = imzml_mode

    @property
    def imzml_mode(self):
        return self._imzml_mode

    # Just for the sake of a clean api this does not really belong here...
    # def add_spectrum(
    #    self,
    #    mz_arr: np.ndarray,
    #    int_arr: np.ndarray,
    #    coordinates: tuple[int, int] | tuple[int, int, int],
    # ):
    #    self._mz_arr.append(mz_arr)
    #    self._int_arr.append(int_arr)
    #    self._coordinates.append(coordinates)

    @contextmanager
    def writer(self):
        yield _Writer(self)

    def to_read_file(self) -> RamReadFile:
        # TODO this method is not really part of the general interface, but is required for this class to be useful
        return RamReadFile(
            mz_arr_list=self._mz_arr_list.copy(),
            int_arr_list=self._int_arr_list.copy(),
            coordinates=self._coordinates_list.copy(),
        )


class _Writer:
    def __init__(self, file: RamWriteFile) -> None:
        self._file = file

    def add_spectrum(self, mz_arr: np.ndarray, int_arr: np.ndarray, coordinates) -> None:
        self._file._mz_arr_list.append(mz_arr)
        self._file._int_arr_list.append(int_arr)
        self._file._coordinates_list.append(coordinates)

    def copy_spectra(self, reader, spectra_indices: Sequence[int], tqdm_position: int | None = None) -> None:
        # TODO reuse the implementation from ImzmlWriter as this is 100% identical
        if tqdm_position is not None:

            def progress_fn(x):
                return tqdm(x, desc=" spectrum", position=tqdm_position)

        else:

            def progress_fn(x):
                return x

        for i_spectrum in progress_fn(spectra_indices):
            mz_arr, int_arr = reader.get_spectrum(i_spectrum)
            self.add_spectrum(mz_arr, int_arr, reader.coordinates[i_spectrum])
