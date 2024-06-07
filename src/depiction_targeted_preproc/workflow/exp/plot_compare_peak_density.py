from pathlib import Path
from typing import Annotated

import polars as pl
import typer
import vegafusion
from typer import Option, Argument

from depiction_targeted_preproc.workflow.qc.plot_peak_density import plot_density_combined_full


def exp_plot_compare_peak_density(
    tables_marker_distances_calib: Annotated[list[Path], Argument()],
    table_marker_distance_uncalib: Annotated[Path, Option()],
    output_pdf: Annotated[Path, Option()],
) -> None:
    vegafusion.enable()

    table = pl.concat(
        [pl.read_parquet(path).with_columns(variant=pl.lit(path.parents[1].name)) for path in
         tables_marker_distances_calib])
    table = pl.concat([table, pl.read_parquet(table_marker_distance_uncalib).with_columns(variant=pl.lit("uncalib"))])

    plot_density_combined_full(df_peak_dist=table, out_pdf=output_pdf)


if __name__ == "__main__":
    typer.run(exp_plot_compare_peak_density)
