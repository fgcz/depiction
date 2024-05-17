from typing import Optional
import scipy
import numpy as np


class EvaluateGaussianSmoothing:
    """
    Evaluates Gaussian smoothing of intensities.
    """

    def __init__(self, window: int = 5, sd: Optional[float] = None):
        """
        :param window: The window size of the Gaussian filter. Default is 5.
        :param sd: The standard deviation of the Gaussian filter. Default is window / 4.
        """
        self._window = window
        self._sd = sd

    def evaluate(self, mz_values: np.ndarray, int_values: np.ndarray) -> np.ndarray:
        """Filters int_array with a Gaussian filter of width window and standard deviation sd."""
        sd = self._sd if self._sd is not None else self._window / 4
        gaussian_filter = scipy.signal.windows.gaussian(self._window, sd)
        gaussian_filter /= np.sum(gaussian_filter)

        if len(int_values) < len(gaussian_filter):
            # Don't filter
            return int_values
        else:
            # the problem with below's implementation was, that e.g. for input of size 1 it would create an output of size 5 (since it choses the bigger size of the two arrays)
            # return scipy.signal.convolve(int_array, gaussian_filter, mode='same')
            values = np.convolve(int_values, gaussian_filter, mode="same")
            values[0] = int_values[0]
            values[-1] = int_values[-1]
            return values
