from pathlib import Path
from typing import Annotated

import polars as pl
import typer
from typer import Option


def exp_mass_list_preparation(
    input_csv_path: Annotated[Path, Option()],
    out_calibration_csv_path: Annotated[Path, Option()],
    out_standards_csv_path: Annotated[Path, Option()],
    out_visualization_csv_path: Annotated[Path, Option()],
) -> None:
    input_df = pl.read_csv(input_csv_path)

    # rename cols
    input_df = input_df.rename({"Marker": "label", "PC-MT (M+H)+": "mass"}).drop("No.")

    # add tol column
    visualization_df = input_df.with_columns(tol=pl.lit(0.25))

    # for the calibration remove the CHCA peaks, they have names starting with CHCA
    calibration_df = visualization_df.filter(~pl.col("name").str.starts_with("CHCA"))

    # for the standards csv only keep the "standard" peaks
    standards_df = visualization_df.filter(pl.col("name").str.to_lowercase().contains("standard"))

    # write the results
    calibration_df.write_csv(out_calibration_csv_path)
    standards_df.write_csv(out_standards_csv_path)
    visualization_df.write_csv(out_visualization_csv_path)


if __name__ == "__main__":
    typer.run(exp_mass_list_preparation)
