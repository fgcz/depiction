from pathlib import Path

import cyclopts
import numpy as np
import polars as pl
import xarray as xr
import yaml

from depiction.image import MultiChannelImage
from depiction.tools.calibrate.calibrate import get_calibration_instance
from depiction.tools.calibrate.config import CalibrationConfig

app = cyclopts.App()


@app.default
def vis_test_mass_shifts(
    calib_hdf5_path: Path, mass_list_path: Path, config_path: Path, output_hdf5_path: Path
) -> None:
    # load inputs
    model_coefs = MultiChannelImage.read_hdf5(calib_hdf5_path, group="model_coefs")
    config = CalibrationConfig.model_validate(yaml.safe_load(config_path.read_text()))

    mass_list = pl.read_csv(mass_list_path)
    calibration = get_calibration_instance(config=config, mass_list=mass_list_path)

    # define test masses
    # to keep it simple for now only 1
    test_masses = np.array([(mass_list["mass"].max() + mass_list["mass"].min()) / 2])
    test_masses_int = np.ones_like(test_masses)

    # compute the shifts
    def compute_shifts(coef):
        result = calibration.apply_spectrum_model(
            spectrum_mz_arr=test_masses, spectrum_int_arr=test_masses_int, model_coef=xr.DataArray(coef, dims=["c"])
        )
        # result = test_mass - shift => shift = test_mass - result
        return xr.DataArray(test_masses - result[0], dims=["m"])

    shifts = xr.apply_ufunc(
        compute_shifts,
        model_coefs.data_spatial,
        input_core_dims=[["c"]],
        output_core_dims=[["m"]],
        vectorize=True,
    ).rename({"m": "c"})
    test_mass_labels = [f"{mass:.2f}" for mass in test_masses]
    shifts_2d = shifts.assign_coords(c=test_mass_labels)

    # save the result
    shifts_img = MultiChannelImage(shifts_2d, is_foreground=model_coefs.fg_mask)
    shifts_img.write_hdf5(output_hdf5_path)


if __name__ == "__main__":
    app()
