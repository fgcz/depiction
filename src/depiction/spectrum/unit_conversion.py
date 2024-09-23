from __future__ import annotations

from enum import Enum
from typing import Self, Annotated

import numpy as np
from pydantic import BaseModel, model_validator, Field


class WindowUnit(Enum):
    mz = "mz"
    index = "index"
    ppm = "ppm"


class WindowSize(BaseModel):
    size: Annotated[int | float, Field(gt=0)]
    unit: WindowUnit

    @model_validator(mode="after")
    def validate_when_index_unit_is_integer(self) -> Self:
        if self.unit == WindowUnit.index:
            if not isinstance(self.size, int):
                raise ValueError(f"Window size must be an integer for unit {self.unit!r}")
        return self

    def convert_to_index_scalar(self, mz_arr: np.ndarray) -> int:
        """Converts the window size to a single scalar value in index units.
        Many functions require a single scalar value for the window size, but if your function supports passing a
        different value for each mz value, you can use the `convert_to_index_array` method instead as it will be more
        accurate.
        """
        if self.unit == WindowUnit.index:
            return self.size
        elif self.unit == WindowUnit.mz:
            mean_mz_diff = np.mean(np.diff(mz_arr))
            return max(round(self.size / mean_mz_diff), 1)
        elif self.unit == WindowUnit.ppm:
            ppm_values = self._compute_ppm_values(mz_arr=mz_arr)
            return max(round(self.size / np.mean(ppm_values)), 1)

    def convert_to_index_array(self, mz_arr: np.ndarray) -> np.ndarray:
        """Converts the window size to an array of scalar values in index units.

        When possible, it is recommended to use this method instead of `convert_to_index_scalar` as it will provide a
        more accurate window size for each mz value.
        """
        if self.unit == WindowUnit.index:
            return np.full(len(mz_arr), fill_value=self.size)
        elif self.unit == WindowUnit.mz:
            diff_arr = np.diff(mz_arr)
            diff_arr = np.append(diff_arr, diff_arr[-1])
            return np.maximum(np.round(self.size / diff_arr).astype(int), 1)
        elif self.unit == WindowUnit.ppm:
            ppm_values = self._compute_ppm_values(mz_arr=mz_arr)
            return np.maximum(np.round(self.size / ppm_values).astype(int), 1)

    def _compute_ppm_values(self, mz_arr: np.ndarray) -> np.ndarray:
        """Returns the ppm values for the given mz array.
        TODO passing the mz_arr for ppm estimation a bit restrictive, in cases where ppm values are known but the
             original mz_arr is not.
        """
        return np.abs(np.diff(mz_arr) / mz_arr[:-1] * 1e6)
