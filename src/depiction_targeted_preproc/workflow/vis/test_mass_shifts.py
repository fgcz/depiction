from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer
import xarray as xr
from typer import Option

from depiction_targeted_preproc.pipeline_config.model import PipelineParameters
from depiction_targeted_preproc.workflow.proc.calibrate import get_calibration_from_config


def vis_test_mass_shifts(
    calib_hdf5_path: Annotated[Path, Option()],
    mass_list_path: Annotated[Path, Option()],
    config_path: Annotated[Path, Option()],
    output_hdf5_path: Annotated[Path, Option()],
) -> None:
    # load inputs
    model_coefs = xr.open_dataarray(calib_hdf5_path, group="model_coefs")
    config = PipelineParameters.parse_yaml(config_path)
    mass_list = pl.read_csv(mass_list_path)
    calibration = get_calibration_from_config(mass_list=mass_list, calib_config=config.calibration)

    # define test masses
    # to keep it simple for now only 1
    test_masses = np.array([(mass_list["mass"].max() + mass_list["mass"].min()) / 2])
    test_masses_int = np.ones_like(test_masses)

    # compute the shifts
    def compute_shifts(coef):
        result = calibration.apply_spectrum_model(spectrum_mz_arr=test_masses, spectrum_int_arr=test_masses_int,
                                                  model_coef=xr.DataArray(coef, dims=["c"]))
        return xr.DataArray(result[0] - test_masses, dims=["m"])

    shifts = xr.apply_ufunc(
        compute_shifts,
        model_coefs,
        input_core_dims=[["c"]],
        output_core_dims=[["m"]],
        vectorize=True,
    ).rename({"m": "c"})
    shifts = shifts.assign_coords(c=test_masses)

    shifts_2d = shifts.set_xindex(["x", "y"]).unstack("i")
    shifts_2d.attrs["bg_value"] = np.nan

    # save the result
    shifts_2d.to_netcdf(output_hdf5_path)


if __name__ == "__main__":
    typer.run(vis_test_mass_shifts)
