from typing import Any, NoReturn

import numpy as np
from numpy._typing import NDArray

from depiction.persistence import ImzmlModeEnum


class RamReader:
    def __init__(self, mz_arr_list, int_arr_list, coordinates) -> None:
        self._mz_arr_list = mz_arr_list
        self._int_arr_list = int_arr_list
        self._coordinates = coordinates

    @property
    def imzml_path(self) -> str:
        print("Warning: imzml_path is not available for RamReadFile")
        return "/dev/null"

    @property
    def ibd_path(self) -> str:
        print("Warning: ibd_path is not available for RamReadFile")
        return "/dev/null"

    @property
    def ibd_mmap(self) -> NoReturn:
        # TODO a problem, right?
        raise NotImplementedError

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self) -> None:
        # do nothing
        pass

    @property
    def imzml_mode(self) -> ImzmlModeEnum:
        # if all mz_arr are identical, then it is 'CONTINUOUS'
        # otherwise, it is 'PROCESSED'
        is_continuous = True
        if len(self._mz_arr_list) > 1:
            is_continuous = True
            for i in range(1, len(self._mz_arr_list)):
                if not np.array_equal(self._mz_arr_list[i], self._mz_arr_list[0]):
                    is_continuous = False
                    break
        return ImzmlModeEnum.CONTINUOUS if is_continuous else ImzmlModeEnum.PROCESSED

    @property
    def n_spectra(self) -> int:
        return len(self._mz_arr_list)

    @property
    def coordinates(self):
        return self._coordinates

    @property
    def coordinates_2d(self):
        return self._coordinates[:, :2]

    def get_spectrum(self, i_spectrum: int) -> tuple[np.ndarray, np.ndarray]:
        return self._mz_arr_list[i_spectrum], self._int_arr_list[i_spectrum]

    def get_spectra(self, i_spectra: list[int]) -> tuple[np.ndarray | list[np.ndarray], np.ndarray | list[np.ndarray]]:
        mz_arr_list = [self._mz_arr_list[i] for i in i_spectra]
        int_arr_list = [self._int_arr_list[i] for i in i_spectra]
        if len({mz_arr.shape for mz_arr in mz_arr_list}) == 1:
            return np.stack(mz_arr_list, axis=0), np.stack(int_arr_list, axis=0)
        else:
            return mz_arr_list, int_arr_list

    def get_spectrum_mz(self, i_spectrum: int) -> NDArray[float]:
        return self._mz_arr_list[i_spectrum]

    def get_spectrum_int(self, i_spectrum) -> NDArray[float]:
        return self._int_arr_list[i_spectrum]

    def get_spectrum_n_points(self, i_spectrum: int) -> int:
        """Returns the number of data points in the i-th spectrum."""
        return len(self._mz_arr_list[i_spectrum])

    def get_spectrum_metadata(self, i_spectrum: int) -> dict[str, Any]:
        return {
            "i_spectrum": i_spectrum,
            "coordinates": self.coordinates[i_spectrum],
        }

    def get_spectra_metadata(self, i_spectra: list[int]) -> list[dict]:
        return [self.get_spectrum_metadata(i) for i in i_spectra]

    def get_spectra_mz_range(self, i_spectra: list[int]) -> tuple[float, float]:
        mz_arr_list = [self._mz_arr_list[i] for i in i_spectra]
        return min([mz_arr.min() for mz_arr in mz_arr_list]), max([mz_arr.max() for mz_arr in mz_arr_list])
