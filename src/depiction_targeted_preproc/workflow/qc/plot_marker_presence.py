from pathlib import Path
from typing import Annotated

import altair as alt
import numpy as np
import polars as pl
import typer
from typer import Option


def plot_marker_presence(df_peak_dist: pl.DataFrame, n_spectra: int, out_path: Path, layout_vertical:bool) -> None:
    # Add a `max_dist` column to the dataframe, that indicates the first bin a particular item falls into
    df = df_peak_dist.with_columns(abs_dist=pl.col("dist").abs()).sort("abs_dist")
    df_cutoffs = pl.DataFrame({"max_dist": [0.05, 0.1, 0.2, 0.3, 0.4, np.inf]}).sort("max_dist")
    df = df.join_asof(df_cutoffs, left_on="abs_dist", right_on="max_dist", strategy="forward").filter(
        pl.col("max_dist").is_finite()
    )

    # For every (label, variant, i_spectrum) compute the minimal `max_dist` value, i.e. the earliest threshold
    df = df.group_by(["label", "variant", "i_spectrum"]).agg(detection_dist=pl.min("max_dist"))

    # Aggregate into a fraction
    df = df.group_by(["label", "variant", "detection_dist"]).agg(fraction=pl.n_unique("i_spectrum") / n_spectra)
    print(df)

    # sort labels by the calibrated image's detection_dist
    sorted_labels = df.filter(detection_dist=df_cutoffs["max_dist"][0], variant="calibrated").sort(
        "fraction", descending=True
    )["label"]

    layout_config = {"column": "variant:N"} if not layout_vertical else {"row": "variant:N"}
    c = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("sum(fraction):Q", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("label:N", sort=list(sorted_labels)),
            color=alt.Color("detection_dist:N"),
            **layout_config
        )
        .properties(
            title="Fraction of spectra with peaks detected per marker at different max distance cutoffs", width=600
        )
    )
    c.save(out_path)


def qc_plot_marker_presence(
    table_marker_distances_baseline: Annotated[Path, Option()],
    table_marker_distances_calib: Annotated[Path, Option()],
    output_pdf: Annotated[Path, Option()],
    layout_vertical: Annotated[bool, Option(is_flag=True)] = False,
) -> None:
    table_calib = pl.read_parquet(table_marker_distances_calib)
    table_baseline = pl.read_parquet(table_marker_distances_baseline)
    table = pl.concat(
        [
            table_calib.with_columns(variant=pl.lit("calibrated")),
            table_baseline.with_columns(variant=pl.lit("baseline_adj")),
        ]
    )
    plot_marker_presence(df_peak_dist=table, n_spectra=table["i_spectrum"].n_unique(), out_path=output_pdf, layout_vertical=layout_vertical)


if __name__ == "__main__":
    typer.run(qc_plot_marker_presence)
