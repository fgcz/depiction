from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, Self, Protocol

if TYPE_CHECKING:
    from types import TracebackType
    from numpy.typing import NDArray

from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

from depiction.persistence.imzml.imzml_mode_enum import ImzmlModeEnum

if TYPE_CHECKING:
    from collections.abc import Sequence
    import numpy as np


# TODO better name


class GenericReader(Protocol):
    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: TracebackType | None
    ) -> None:
        self.close()

    def close(self) -> None: ...

    @property
    def imzml_mode(self) -> ImzmlModeEnum:
        """Returns the mode of the imzML file."""
        ...

    @property
    def n_spectra(self) -> int:
        """The number of spectra available in the .imzML file."""
        ...

    @cached_property
    def coordinates(self) -> NDArray[int]:
        """Returns the coordinates of the spectra in the imzML file, shape (n_spectra, n_dim)."""
        ...

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
        ...

    def get_spectrum_int(self, i_spectrum: int) -> NDArray[float]:
        """Returns the intensity values of the i-th spectrum."""
        ...

    def get_spectrum_coordinates(self, i_spectrum: int) -> NDArray[int]:
        """Returns the coordinates of the i-th spectrum."""
        return self.coordinates[i_spectrum]

    def get_spectrum_n_points(self, i_spectrum: int) -> int:
        """Returns the number of data points in the i-th spectrum."""
        return len(self.get_spectrum_mz(i_spectrum))

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


class GenericWriter(Protocol):

    # TODO this currently does not impl __enter__ and __exit__ as GenericReader

    @classmethod
    def open(cls, path: str | Path, imzml_mode: ImzmlModeEnum, imzml_alignment_tracking: bool = True) -> Self:
        """Opens an imzML file."""
        ...

    def close(self) -> None: ...

    @property
    def imzml_mode(self) -> ImzmlModeEnum:
        """Returns the mode of the imzML file."""
        ...

    def add_spectrum(
        self,
        mz_arr: np.ndarray,
        int_arr: np.ndarray,
        coordinates: tuple[int, int] | tuple[int, int, int],
    ) -> None: ...

    def copy_spectra(
        self,
        reader: GenericReader,
        spectra_indices: Sequence[int],
        tqdm_position: int | None = None,
    ) -> None:
        """Copies spectra from an existing reader. Not optimized yet.
        :param reader: The reader to copy from.
        :param spectra_indices: The indices of the spectra to copyl.
        """
        if tqdm_position is not None:

            def progress_fn(x: Sequence[int]) -> tqdm:
                return tqdm(x, desc=" spectrum", position=tqdm_position)

        else:

            def progress_fn(x: Sequence[int]) -> Sequence[int]:
                return x

        for spectrum_index in progress_fn(spectra_indices):
            mz_arr, int_arr, coordinates = reader.get_spectrum_with_coords(spectrum_index)
            self.add_spectrum(mz_arr, int_arr, coordinates)
