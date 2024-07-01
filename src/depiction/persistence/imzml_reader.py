from __future__ import annotations

import mmap
from functools import cached_property
from typing import Any, TYPE_CHECKING

import numpy as np
import pyimzml.ImzMLParser

from depiction.persistence.imzml_mode_enum import ImzmlModeEnum

if TYPE_CHECKING:
    from types import TracebackType
    from pathlib import Path
    from numpy.typing import NDArray


class ImzmlReader:
    """
    Memmap based reader for imzML files, that can be pickled.
    """

    def __init__(
        self,
        mz_arr_offsets: list[int],
        mz_arr_lengths: list[int],
        mz_arr_dtype: str,
        int_arr_offsets: list[int],
        int_arr_lengths: list[int],
        int_arr_dtype: str,
        coordinates: NDArray[int],
        imzml_path: Path,
    ) -> None:
        self._imzml_path = imzml_path
        self._ibd_file = None
        self._ibd_mmap = None

        self._mz_arr_offsets = mz_arr_offsets
        self._mz_arr_lengths = mz_arr_lengths
        self._mz_arr_dtype = mz_arr_dtype
        self._int_arr_offsets = int_arr_offsets
        self._int_arr_lengths = int_arr_lengths
        self._int_arr_dtype = int_arr_dtype
        self._coordinates = coordinates

        self._mz_bytes = np.dtype(mz_arr_dtype).itemsize
        self._int_bytes = np.dtype(int_arr_dtype).itemsize

    def __getstate__(self) -> dict[str, Any]:
        return {
            "imzml_path": self._imzml_path,
            "mz_arr_offsets": self._mz_arr_offsets,
            "mz_arr_lengths": self._mz_arr_lengths,
            "mz_arr_dtype": self._mz_arr_dtype,
            "int_arr_offsets": self._int_arr_offsets,
            "int_arr_lengths": self._int_arr_lengths,
            "int_arr_dtype": self._int_arr_dtype,
            "mz_bytes": self._mz_bytes,
            "int_bytes": self._int_bytes,
            "coordinates": self._coordinates,
        }

    # TODO
    def __setstate__(self, state: dict[str, Any]) -> None:
        # self._portable_reader = state["portable_reader"]
        self._imzml_path = state["imzml_path"]
        self._ibd_file = None
        self._ibd_mmap = None
        self._mz_arr_offsets = state["mz_arr_offsets"]
        self._mz_arr_lengths = state["mz_arr_lengths"]
        self._mz_arr_dtype = state["mz_arr_dtype"]
        self._int_arr_offsets = state["int_arr_offsets"]
        self._int_arr_lengths = state["int_arr_lengths"]
        self._int_arr_dtype = state["int_arr_dtype"]
        self._mz_bytes = state["mz_bytes"]
        self._int_bytes = state["int_bytes"]
        self._coordinates = state["coordinates"]

    @property
    def imzml_path(self) -> Path:
        """The path to the .imzML file."""
        return self._imzml_path

    @property
    def ibd_path(self) -> Path:
        """The path to the .ibd file."""
        return self._imzml_path.with_suffix(".ibd")

    @property
    def ibd_mmap(self) -> mmap.mmap:
        """The mmap object for the .ibd file. This will open a file handle if necessary."""
        if self._ibd_mmap is None:
            self._ibd_file = self.ibd_path.open("rb")
            self._ibd_mmap = mmap.mmap(
                fileno=self._ibd_file.fileno(),
                length=0,
                access=mmap.ACCESS_READ,
            )
        return self._ibd_mmap

    def __enter__(self) -> ImzmlReader:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        self.close()

    def close(self) -> None:
        """Closes the .ibd file handles, if open."""
        if self._ibd_mmap is not None:
            self._ibd_mmap.close()
            self._ibd_mmap = None
        if self._ibd_file is not None:
            self._ibd_file.close()
            self._ibd_file = None

    @cached_property
    def imzml_mode(self) -> ImzmlModeEnum:
        """Returns the mode of the imzML file."""
        # maybe this can be solved more elegantly in the future, but right now this works (if all offsets are identical,
        # then we know it's CONTINUOUS)
        if len({*self._mz_arr_offsets}) == 1:
            return ImzmlModeEnum.CONTINUOUS
        else:
            return ImzmlModeEnum.PROCESSED

    @property
    def n_spectra(self) -> int:
        """The number of spectra available in the .imzML file."""
        return len(self._int_arr_lengths)

    @cached_property
    def coordinates(self) -> NDArray[int]:
        """Returns the coordinates of the spectra in the imzML file, shape (n_spectra, n_dim)."""
        return self._coordinates

    @cached_property
    def coordinates_2d(self) -> NDArray[int]:
        """Returns the coordinates of the spectra in the imzML file, shape (n_spectra, 2)."""
        return self.coordinates[:, :2]

    def get_spectrum(self, i_spectrum: int) -> tuple[NDArray[float], NDArray[float]]:
        """Returns the m/z and intensity arrays of the i-th spectrum."""
        return self.get_spectrum_mz(i_spectrum=i_spectrum), self.get_spectrum_int(i_spectrum=i_spectrum)

    def get_spectrum_with_coords(self, i_spectrum: int) -> tuple[NDArray[float], NDArray[float], NDArray[float]]:
        """Returns the m/z, intensity and v arrays of the i-th spectrum."""
        mz_arr = self.get_spectrum_mz(i_spectrum=i_spectrum)
        int_arr = self.get_spectrum_int(i_spectrum=i_spectrum)
        coords = self.get_spectrum_coordinates(i_spectrum=i_spectrum)
        return mz_arr, int_arr, coords

    def get_spectra(
        self, i_spectra: list[int]
    ) -> tuple[NDArray[float] | list[NDArray[float]], NDArray[float] | list[NDArray[float]]]:
        """Returns the m/z and intensity arrays of the specified spectra.
        For continuous mode, the arrays are stacked into a single array, whereas
        for processed mode, a list of arrays is returned as they might not have
        the same shape.
        """
        if self.imzml_mode == ImzmlModeEnum.CONTINUOUS:
            mz_arr = self.get_spectrum_mz(i_spectrum=i_spectra[0])
            mz_arr_list = np.repeat(mz_arr[np.newaxis, :], len(i_spectra), axis=0)
            int_arr_list = np.stack([self.get_spectrum_int(i_spectrum=i) for i in i_spectra], axis=0)
            return mz_arr_list, int_arr_list
        else:
            return tuple(zip(*[self.get_spectrum(i_spectrum=i) for i in i_spectra]))

    def get_spectrum_mz(self, i_spectrum: int) -> NDArray[float]:
        """Returns the m/z values of the i-th spectrum."""
        file = self.ibd_mmap
        file.seek(self._mz_arr_offsets[i_spectrum])
        mz_bytes = file.read(self._mz_arr_lengths[i_spectrum] * self._mz_bytes)
        return np.frombuffer(mz_bytes, dtype=self._mz_arr_dtype)

    def get_spectrum_int(self, i_spectrum: int) -> NDArray[float]:
        """Returns the intensity values of the i-th spectrum."""
        file = self.ibd_mmap
        file.seek(self._int_arr_offsets[i_spectrum])
        int_bytes = file.read(self._int_arr_lengths[i_spectrum] * self._int_bytes)
        return np.frombuffer(int_bytes, dtype=self._int_arr_dtype)

    def get_spectrum_coordinates(self, i_spectrum: int) -> NDArray[int]:
        """Returns the coordinates of the i-th spectrum."""
        return self.coordinates[i_spectrum]

    def get_spectrum_n_points(self, i_spectrum: int) -> int:
        """Returns the number of data points in the i-th spectrum."""
        return self._int_arr_lengths[i_spectrum]

    def get_spectra_mz_range(self, i_spectra: list[int] | None) -> tuple[float, float]:
        """Returns the m/z range of the given spectra, returning the global min and max m/z value."""
        if i_spectra is None:
            i_spectra = range(self.n_spectra)
        mz_min = np.inf
        mz_max = -np.inf
        for i_spectrum in i_spectra:
            mz_arr = self.get_spectrum_mz(i_spectrum)
            mz_min = mz_arr[0] if mz_arr[0] < mz_min else mz_min
            mz_max = mz_arr[-1] if mz_arr[-1] > mz_max else mz_max
        return mz_min, mz_max

    @classmethod
    def parse_imzml(cls, path: Path) -> ImzmlReader:
        """Parses an imzML file and returns an ImzmlReader."""
        with pyimzml.ImzMLParser.ImzMLParser(path) as parser:
            portable_reader = parser.portable_spectrum_reader()
        return ImzmlReader(
            mz_arr_offsets=portable_reader.mzOffsets,
            mz_arr_lengths=portable_reader.mzLengths,
            mz_arr_dtype=portable_reader.mzPrecision,
            int_arr_offsets=portable_reader.intensityOffsets,
            int_arr_lengths=portable_reader.intensityLengths,
            int_arr_dtype=portable_reader.intensityPrecision,
            coordinates=np.asarray(portable_reader.coordinates),
            imzml_path=path,
        )

    def __str__(self) -> str:
        return (
            f"ImzmlReader[{self._imzml_path}, n_spectra={self.n_spectra},"
            f" int_arr_dtype={self._int_arr_dtype}, mz_arr_dtype={self._mz_arr_dtype}]"
        )
