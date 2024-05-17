from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import seaborn
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from ion_mapper.evaluate_bins import EvaluateBins
from ion_mapper.evaluate_mean_spectrum import EvaluateMeanSpectrum
from ion_mapper.parallel_ops import ParallelConfig
from ion_mapper.persistence import ImzmlReadFile


class VisualizeCalibrationTargets:
    def __init__(self, *, mean_mz_arr: NDArray[float], mean_int_arr: NDArray[float]):
        self._mean_mz_arr = mean_mz_arr
        self._mean_int_arr = mean_int_arr

    @classmethod
    def from_mean_spectrum(
        cls, mean_mz_arr: NDArray[float], mean_int_arr: NDArray[float],
    ) -> VisualizeCalibrationTargets:
        return cls(mean_mz_arr=mean_mz_arr, mean_int_arr=mean_int_arr)

    @classmethod
    def from_imzml_file(cls, read_file: ImzmlReadFile, parallel_config: ParallelConfig) -> VisualizeCalibrationTargets:
        mean_mz_arr, mean_int_arr = cls._get_mean_spectrum_for_file(
            read_file=read_file, parallel_config=parallel_config,
        )
        return cls(mean_mz_arr=mean_mz_arr, mean_int_arr=mean_int_arr)

    def get_peak_surrounding_series(
        self,
        mz_center: float,
        vis_tol: float,
    ) -> pd.Series:
        # find the relevant indices in the mean_mz_arr
        left_idx = np.searchsorted(self._mean_mz_arr, mz_center - vis_tol, side="left")
        right_idx = np.searchsorted(self._mean_mz_arr, mz_center + vis_tol, side="right")

        # get the relevant data
        mz_arr = self._mean_mz_arr[left_idx:right_idx]
        int_arr = self._mean_int_arr[left_idx:right_idx]

        # fill in the values at the edges by linear interpolation
        if mz_arr[0] > mz_center - vis_tol:
            int_value = np.interp(mz_center - vis_tol, mz_arr, int_arr)
            mz_arr = np.concatenate([[mz_center - vis_tol], mz_arr])
            int_arr = np.concatenate([[int_value], int_arr])
        if mz_arr[-1] < mz_center + vis_tol:
            int_value = np.interp(mz_center + vis_tol, mz_arr, int_arr)
            mz_arr = np.concatenate([mz_arr, [mz_center + vis_tol]])
            int_arr = np.concatenate([int_arr, [int_value]])

        # create the dataframe
        return pd.Series({"m/z": mz_arr, "intensity": int_arr, "mz_center": mz_center})

    def plot_target_peak_surrounding(
        self,
        mz_center: float,
        mz_tol: float,
        vis_tol: float,
        ax: plt.Axes,
        title: str,
    ):
        data = self.get_peak_surrounding_series(mz_center=mz_center, vis_tol=vis_tol).explode(["m/z", "intensity"])

        seaborn.lineplot(data=data, x="m/z", y="intensity", ax=ax, label="mean spectrum")

        ax.plot(*eval_range(mean_mz, mean_int, [mz_center - vis_tol, mz_center + vis_tol]))
        ax.set_title(title)
        ax.fill_between(
            *eval_range(mean_mz, mean_int, [mz_center - mz_tol, mz_center + mz_tol]), alpha=0.5, label="signal",
        )
        ax.axvline(mz_center, color="red", linestyle="-")
        ax.axvline(mz_center - mz_tol, color="gray", linestyle="--")
        ax.axvline(mz_center + mz_tol, color="gray", linestyle="--")
        ax.set_yscale("log")

    def plot_target_peak_grid(
        self,
        target_mzs: NDArray[float],
        target_labels: list[str],
        mz_tol: NDArray[float],
        sort_by_mz: bool = True,
        title: str | None = None,
        grid_cols: int = 5,
    ):
        if sort_by_mz:
            sort_idx = np.argsort(target_mzs)
            target_mzs = target_mzs[sort_idx]
            target_labels = [target_labels[i] for i in sort_idx]
            mz_tol = mz_tol[sort_idx]

        grid_rows = int(np.ceil(len(target_mzs) / grid_cols))

        fig, axs = plt.subplots(
            grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3), sharey=True, squeeze=False,
        )
        for i_row, i_col in itertools.product(range(grid_rows), range(grid_cols)):
            i_label = i_row * grid_cols + i_col
            if i_label >= len(target_mzs):
                break
            mz_center = target_mzs[i_label]

            self.plot_target_peak_surrounding(
                mz_center=mz_center,
                mz_tol=mz_tol[i_label],
                vis_tol=mz_tol[i_label] + 1.0,
                ax=axs[i_row, i_col],
                title=target_labels[i_label],
            )

        if title:
            fig.suptitle(title)
        fig.tight_layout()

        return fig, axs

    @classmethod
    def _get_mean_spectrum_for_file(
        cls, read_file: ImzmlReadFile, parallel_config: ParallelConfig,
    ) -> tuple[NDArray[float], NDArray[float]]:
        # TODO technically, this method does not really belong here, but for now it is fine.
        with read_file.reader() as reader:
            mz_arr_first = reader.get_spectrum_mz(0)
        eval_bins = EvaluateBins.from_mz_values(mz_arr_first)
        evaluate_mean = EvaluateMeanSpectrum(parallel_config=parallel_config, eval_bins=eval_bins)
        return evaluate_mean.evaluate_file(read_file)
