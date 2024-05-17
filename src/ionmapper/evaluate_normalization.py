import numpy as np

# TODO deprecated, use tools.normalize_intensities instead

class EvaluateTICNormalization:
    """
    Evaluates the TIC normalization of spectra.
    Since some software does not normalize to 1, this can be specified with the target_value parameter.
    """

    def __init__(self, target_value: float = 1.0) -> None:
        """
        :param target_value: The target value to normalize to. Default is 1, i.e. the sum of all intensities is 1.
        """
        self._target_value = target_value

    def evaluate(self, mz_values: np.ndarray, int_values: np.ndarray) -> np.ndarray:
        """Evaluates the TIC normalization for the provided spectrum of m/z and intensity values."""
        tic = np.abs(int_values).sum()
        int_values = int_values * (self._target_value / tic)
        return int_values


class EvaluateMedianNormalization:
    """Evaluates the median normalization of spectra."""

    def __init__(self, target_value: float = 1.0) -> None:
        """
        :param target_value: The target value to normalize to. Default is 1, i.e. the median of all intensities is 1.
        """
        self._target_value = target_value

    def evaluate(self, mz_values: np.ndarray, int_values: np.ndarray) -> np.ndarray:
        """Evaluates the median normalization for the provided spectrum of m/z and intensity values."""
        return int_values * (self._target_value / np.median(int_values))
