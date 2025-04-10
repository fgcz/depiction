import numpy as np
from pydantic import BaseModel


def _resample(mz_arr, int_arr):
    min_step = np.min(np.diff(mz_arr))
    new_mz_arr = np.arange(mz_arr[0], mz_arr[-1], min_step)
    new_int_arr = np.interp(new_mz_arr, mz_arr, int_arr)
    return new_int_arr


def estimate_noise_level_by_histogram(mz_arr, int_arr, eta: float = 3.0, n_bin: int = 30):
    """Estimates signal-to-noise ratio (SNR) by histogram analysis, as described in [1].

    TODO: in the paper they first perform Savitzk-Golay filtering to the raw data, also i don't implement the full method

    [1]: Jia, M.; Wu, M.; Li, Y.; Xiong, B.; Wang, L.; Ling, X.; Cheng, W.; Dong, W.-F.
     Quantitative Method for Liquid Chromatography–Mass Spectrometry Based on Multi-Sliding Window and Noise Estimation.
     Processes 2022, 10, 1098. https://doi.org/10.3390/pr10061098
    """

    int_arr = _resample(mz_arr, int_arr)

    # compute expectation and standard deviation of the dat
    data_exp = np.mean(int_arr)
    data_std = np.std(int_arr)

    # histogram is computed up to a limit only
    hist_limit = data_exp + eta * data_std

    # compute the histogram bins
    hist_bin_edges = np.histogram_bin_edges([0, hist_limit], bins=n_bin)
    hist_bin_counts = np.histogram(int_arr, bins=hist_bin_edges)[0]

    # read out the noise
    median_index = np.argsort(hist_bin_counts)[n_bin // 2 + 1]
    return max(hist_bin_edges[median_index], 1)


class FilterBySnrConfig(BaseModel):
    eta: float = 3.0
    n_bin: int = 30


class FilterBySnr(PeakFilter):
    pass
