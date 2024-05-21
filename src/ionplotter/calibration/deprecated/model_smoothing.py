import numpy as np
import scipy.ndimage
from numpy.typing import NDArray

from ionplotter.calibration.models.linear_model import LinearModel
from ionplotter.calibration.models.polynomial_model import PolynomialModel


class ModelSmoothing:
    """
    Performs some smoothing of the model parameters obtained for a full image.
    """

    def __init__(self, sigma: float) -> None:
        self._sigma = sigma

    def smooth_spatial(
        self, coordinates: NDArray[int], models: list[LinearModel | PolynomialModel]
    ) -> list[LinearModel | PolynomialModel]:
        """
        Smoothes the values in ``values`` using a Gaussian kernel in a spatial neighborhood.
        :param coordinates: shape (n_points, 2) the coordinates of the points
        :param models: the models whose coefficients to smooth
        :return list of smoothed models, with the same type as the input models
        """
        model_type, values = self.get_model_type_and_values(models=models)

        # input validation
        if coordinates.shape[0] != values.shape[0]:
            raise ValueError("coordinates and values must have the same number of rows")
        if coordinates.shape[1] != 2:
            raise ValueError("coordinates must have two columns (2d coordinates)")

        # assign the values to a grid
        value_grid = np.zeros(coordinates.max(axis=0) + 1, dtype=values.dtype)
        value_grid[coordinates[:, 0], coordinates[:, 1]] = values

        # smooth the grid
        smoothed_grid = scipy.ndimage.gaussian_filter(value_grid, sigma=self._sigma, axes=(0, 1))

        # assign the smoothed values back to the points
        smoothed_values = smoothed_grid[coordinates[:, 0], coordinates[:, 1]]

        return [model_type(coef=coef) for coef in smoothed_values]

    def smooth_sequential(self, models: list[LinearModel | PolynomialModel]) -> list[LinearModel | PolynomialModel]:
        """
        Smoothes the values in ``values`` using a Gaussian kernel in a sequential neighborhood.
        :param models: the models whose coefficients to smooth
        :return list of smoothed models, with the same type as the input models
        """
        model_type, values = self.get_model_type_and_values(models=models)
        smoothed_values = scipy.ndimage.gaussian_filter1d(values, sigma=self._sigma, axis=0)
        return [model_type(coef=coef) for coef in smoothed_values]

    def get_model_type_and_values(
        self, models: list[LinearModel | PolynomialModel]
    ) -> tuple[type[LinearModel | PolynomialModel], NDArray[float]]:
        model_type = self._check_model_type(models=models)
        return model_type, np.array([model.coef for model in models])

    def _check_model_type(self, models: list[LinearModel | PolynomialModel]) -> type[LinearModel | PolynomialModel]:
        model_types = {type(model) for model in models}
        if len(model_types) != 1:
            raise ValueError(f"all models must have the same type, found: {model_types}")
        return list(model_types)[0]
