from pathlib import Path

import cyclopts
import xarray

app = cyclopts.App()


@app.default
def export_model_coefs(calib_data_path: Path, output_hdf5_path: Path) -> None:
    calib_data = xarray.open_dataarray(calib_data_path, group="model_coefs")
    calib_data.to_netcdf(output_hdf5_path)


if __name__ == "__main__":
    app()
