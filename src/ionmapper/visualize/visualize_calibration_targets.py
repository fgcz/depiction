from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from ionmapper.evaluate_mean_spectrum import EvaluateMeanSpectrum
from ionmapper.evaluate_bins import EvaluateBins
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ionmapper.persistence import ImzmlReadFile
    from ionmapper.parallel_ops import ParallelConfig
    from numpy.typing import NDArray


class VisualizeCalibrationTargets:
    def __init__(self, *, mean_mz_arr: NDArray[float], mean_int_arr: NDArray[float]) -> None:
        self._mean_mz_arr = mean_mz_arr
        self._mean_int_arr = mean_int_arr

    @property
    def mean_mz_arr(self) -> NDArray[float]:
        return self._mean_mz_arr

    @property
    def mean_int_arr(self) -> NDArray[float]:
        return self._mean_int_arr

    @classmethod
    def from_mean_spectrum(
        cls, mean_mz_arr: NDArray[float], mean_int_arr: NDArray[float]
    ) -> VisualizeCalibrationTargets:
        return cls(mean_mz_arr=mean_mz_arr, mean_int_arr=mean_int_arr)

    @classmethod
    def from_imzml_file(cls, read_file: ImzmlReadFile, parallel_config: ParallelConfig) -> VisualizeCalibrationTargets:
        mean_mz_arr, mean_int_arr = cls._get_mean_spectrum_for_file(
            read_file=read_file, parallel_config=parallel_config
        )
        return cls(mean_mz_arr=mean_mz_arr, mean_int_arr=mean_int_arr)

    def integrate_signal_strengths(self, target_mzs: NDArray[float], mz_tol: float) -> pd.DataFrame:
        collect = defaultdict(list)
        for i_label, target_mz in enumerate(target_mzs):
            arr_min = np.searchsorted(self._mean_mz_arr, target_mz - mz_tol, side="left")
            arr_max = np.searchsorted(self._mean_mz_arr, target_mz + mz_tol, side="right")
            collect["mz"].append(target_mz)
            collect["intensity"].append(self._mean_int_arr[arr_min:arr_max].sum())
            collect["i_label"].append(i_label)
        return pd.DataFrame(collect)

    def select_signal_by_strengths(
        self, target_mzs: NDArray[float], mz_tol: float, n_signals: int = 0, strongest: bool = True
    ) -> pd.DataFrame:
        df = self.integrate_signal_strengths(target_mzs=target_mzs, mz_tol=mz_tol)
        df = df.sort_values(by="intensity", ascending=not strongest)
        if n_signals:
            df = df.head(n_signals)
        return df

    def plot_target_peak_surrounding(
        self,
        mz_center: float,
        mz_tol: float,
        vis_tol: float,
        ax: plt.Axes,
        title: str,
        log_scale: bool = True,
    ) -> None:
        mean_mz = self._mean_mz_arr
        mean_int = self._mean_int_arr

        if vis_tol < mz_tol:
            # Note: mz_tol and vis_tol are separately (i.e. they are not added together)
            raise ValueError("vis_tol should be >= mz_tol")

        def eval_range(x, y, xlim):
            y_lower, y_upper = np.interp(xlim, x, y)
            mask = (x >= xlim[0]) & (x <= xlim[1])
            x_res = np.concatenate([[xlim[0]], x[mask], [xlim[1]]])
            y_res = np.concatenate([[y_lower], y[mask], [y_upper]])
            return x_res, y_res

        ax.plot(*eval_range(mean_mz, mean_int, [mz_center - vis_tol, mz_center + vis_tol]))
        ax.set_title(title)
        ax.fill_between(
            *eval_range(mean_mz, mean_int, [mz_center - mz_tol, mz_center + mz_tol]), alpha=0.5, label="signal"
        )
        ax.axvline(mz_center, color="red", linestyle="-")
        ax.axvline(mz_center - mz_tol, color="gray", linestyle="--")
        ax.axvline(mz_center + mz_tol, color="gray", linestyle="--")
        if log_scale:
            ax.set_yscale("log")

    def plot_target_peak_grid(
        self,
        target_mzs: NDArray[float],
        target_labels: list[str],
        mz_tol: NDArray[float],
        sort_by_mz: bool = True,
        title: str | None = None,
        grid_cols: int = 5,
        log_scale: bool = True,
    ):
        if sort_by_mz:
            sort_idx = np.argsort(target_mzs)
            target_mzs = target_mzs[sort_idx]
            target_labels = [target_labels[i] for i in sort_idx]
            mz_tol = mz_tol[sort_idx]

        grid_rows = int(np.ceil(len(target_mzs) / grid_cols))

        fig, axs = plt.subplots(
            grid_rows, grid_cols, figsize=(grid_cols * 3, grid_rows * 3), sharey=True, squeeze=False
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
                log_scale=log_scale,
            )

        if title:
            fig.suptitle(title)
        fig.tight_layout()

        return fig, axs

    @classmethod
    def _get_mean_spectrum_for_file(
        cls, read_file: ImzmlReadFile, parallel_config: ParallelConfig
    ) -> tuple[NDArray[float], NDArray[float]]:
        with read_file.reader() as reader:
            mz_arr_first = reader.get_spectrum_mz(0)
        eval_bins = EvaluateBins.from_mz_values(mz_arr_first)
        evaluate_mean = EvaluateMeanSpectrum(parallel_config=parallel_config, eval_bins=eval_bins)
        return evaluate_mean.evaluate_file(read_file)
