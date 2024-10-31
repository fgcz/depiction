from __future__ import annotations

import numpy as np
from depiction.calibration.models import LinearModel, PolynomialModel
from numpy.typing import NDArray

ModelType = LinearModel | PolynomialModel


def fit_model(x: NDArray[np.float64], y: NDArray[np.float64], model_type: str) -> ModelType:
    """Fits a model to the given data, with the particular model_type."""
    model: ModelType
    if len(x) < 3:
        # If there are not enough points, return a zero model.
        if model_type.startswith("poly_"):
            model = PolynomialModel.zero()
        elif model_type.startswith("linear"):
            model = LinearModel.zero()
        else:
            raise ValueError(f"Unknown {model_type=}")
    elif model_type == "linear":
        model = LinearModel.fit_lsq(x_arr=x, y_arr=y)
    elif model_type.startswith("poly_"):
        degree = int(model_type.split("_")[1])
        model = PolynomialModel.fit_lsq(x_arr=x, y_arr=y, degree=degree)
    elif model_type == "linear_siegelslopes":
        model = LinearModel.fit_siegelslopes(x_arr=x, y_arr=y)
    else:
        raise ValueError(f"Unknown {model_type=}")
    return model
