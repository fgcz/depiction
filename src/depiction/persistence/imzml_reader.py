from __future__ import annotations

import mmap
from functools import cached_property
from typing import Any, TYPE_CHECKING

import numpy as np

from depiction.persistence.imzml_mode_enum import ImzmlModeEnum

if TYPE_CHECKING:
    from pathlib import Path
    import pyimzml.ImzMLParser
    from numpy.typing import NDArray


class ImzmlReader:
    """
    Memmap based reader for imzML files, that can be pickled.
    TODO: Consider whether to decouple the implementation even further from pyimzml by not using PortableSpectrumReader
          except for conversion. At the moment it is not necessary, but it might be useful for optimizing further in
          the future.
    """

    def __init__(
        self,
        portable_reader: pyimzml.ImzMLParser.PortableSpectrumReader,
        imzml_path: Path,
    ) -> None:
        self._portable_reader = portable_reader
        self._imzml_path = imzml_path
        self._ibd_file = None
        self._ibd_mmap = None

        size_dict = {"f": 4, "d": 8, "i": 4, "l": 8}
        self._mz_bytes = size_dict[portable_reader.mzPrecision]
        self._int_bytes = size_dict[portable_reader.intensityPrecision]

    def __getstate__(self) -> dict[str, Any]:
        return {
            "portable_reader": self._portable_reader,
            "imzml_path": self._imzml_path,
            "mz_bytes": self._mz_bytes,
            "int_bytes": self._int_bytes,
        }

    def __setstate__(self, state: dict[str, Any]):
        self._portable_reader = state["portable_reader"]
        self._imzml_path = state["imzml_path"]
        self._ibd_file = None
        self._ibd_mmap = None
        self._mz_bytes = state["mz_bytes"]
        self._int_bytes = state["int_bytes"]

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
            self._open_ibd_mmap()
        return self._ibd_mmap

    def _open_ibd_mmap(self) -> None:
        # TODO maybe this can be converted into cached_property too
        self._ibd_file = open(self.ibd_path, "rb")
        self._ibd_mmap = mmap.mmap(
            fileno=self._ibd_file.fileno(),
            length=0,
            access=mmap.ACCESS_READ,
        )

    def __enter__(self) -> ImzmlReader:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
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
        if len({*self._portable_reader.mzOffsets}) == 1:
            return ImzmlModeEnum.CONTINUOUS
        else:
            return ImzmlModeEnum.PROCESSED

    @property
    def n_spectra(self) -> int:
        return len(self._portable_reader.mzOffsets)

    @cached_property
    def coordinates(self) -> NDArray[int]:
        """Returns the coordinates of the spectra in the imzML file, shape (n_spectra, n_dim)."""
        return np.asarray(self._portable_reader.coordinates)

    @cached_property
    def coordinates_2d(self) -> NDArray[int]:
        """Returns the coordinates of the spectra in the imzML file, shape (n_spectra, 2)."""
        return self.coordinates[:, :2]

    def get_spectrum(self, i_spectrum: int) -> tuple[NDArray[float], NDArray[float]]:
        """Returns the m/z and intensity arrays of the i-th spectrum."""
        return self._portable_reader.read_spectrum_from_file(self.ibd_mmap, i_spectrum)

    def get_spectrum_with_coords(self, i_spectrum: int) -> tuple[NDArray[float], NDArray[float], NDArray[float]]:
        """Returns the m/z, intensity and v arrays of the i-th spectrum."""
        mz_arr, int_arr = self.get_spectrum(i_spectrum)
        coordinates = self.get_spectrum_coordinates(i_spectrum)
        return mz_arr, int_arr, coordinates

    def get_spectra(
        self, i_spectra: list[int]
    ) -> tuple[NDArray[float] | list[NDArray[float]], NDArray[float] | list[NDArray[float]]]:
        """
        Returns the m/z and intensity arrays of the specified spectra.
        For continuous mode, the arrays are stacked into a single array, whereas
        for processed mode, a list of arrays is returned as they might not have
        the same shape.
        """
        if self.imzml_mode == ImzmlModeEnum.CONTINUOUS:
            mz_arr = self.get_spectrum_mz(i_spectra[0])
            mz_arr_list = np.repeat(mz_arr[np.newaxis, :], len(i_spectra), axis=0)
            int_arr_list = np.stack([self.get_spectrum_int(i) for i in i_spectra], axis=0)
            return mz_arr_list, int_arr_list
        else:
            return tuple(zip(*[self.get_spectrum(i) for i in i_spectra]))

    def get_spectrum_mz(self, i_spectrum: int) -> NDArray[float]:
        """Returns the m/z values of the i-th spectrum."""
        file = self.ibd_mmap
        file.seek(self._portable_reader.mzOffsets[i_spectrum])
        mz_bytes = file.read(self._portable_reader.mzLengths[i_spectrum] * self._mz_bytes)
        return np.frombuffer(mz_bytes, dtype=self._portable_reader.mzPrecision)

    def get_spectrum_int(self, i_spectrum: int) -> NDArray[float]:
        """Returns the intensity values of the i-th spectrum."""
        file = self.ibd_mmap
        file.seek(self._portable_reader.intensityOffsets[i_spectrum])
        int_bytes = file.read(self._portable_reader.intensityLengths[i_spectrum] * self._int_bytes)
        return np.frombuffer(int_bytes, dtype=self._portable_reader.intensityPrecision)

    def get_spectrum_coordinates(self, i_spectrum: int) -> NDArray[int]:
        """Returns the coordinates of the i-th spectrum."""
        return self.coordinates[i_spectrum]

    def get_spectrum_n_points(self, i_spectrum: int) -> int:
        """Returns the number of data points in the i-th spectrum."""
        return self._portable_reader.intensityLengths[i_spectrum]

    def get_spectra_mz_range(self, i_spectra: list[int]) -> tuple[float, float]:
        """Returns the m/z range of the given spectra, returning the global min and max m/z value."""
        mz_min = np.inf
        mz_max = -np.inf
        for i_spectrum in i_spectra:
            mz_arr = self.get_spectrum_mz(i_spectrum)
            mz_min = mz_arr[0] if mz_arr[0] < mz_min else mz_min
            mz_max = mz_arr[-1] if mz_arr[-1] > mz_max else mz_max
        return mz_min, mz_max
