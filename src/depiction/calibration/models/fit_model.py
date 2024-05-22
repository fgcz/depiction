from typing import Union
from numpy.typing import NDArray
from depiction.calibration.models import LinearModel, PolynomialModel


def fit_model(x: NDArray[float], y: NDArray[float], model_type: str) -> Union[LinearModel, PolynomialModel]:
    if len(x) < 3:
        # If there are not enough points, return a zero model.
        if model_type.startswith("poly_"):
            model_class = PolynomialModel
        elif model_type.startswith("linear"):
            model_class = LinearModel
        else:
            raise ValueError(f"Unknown {model_type=}")
        model = model_class.zero()
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
