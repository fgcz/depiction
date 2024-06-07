from pathlib import Path
from typing import Annotated

import altair as alt
import polars as pl
import typer
import vegafusion
from KDEpy import FFTKDE
from typer import Option

from depiction_targeted_preproc.workflow.qc.plot_calibration_map import get_mass_groups


def subsample_dataframe(df: pl.DataFrame) -> pl.DataFrame:
    n_samples = min(len(df), 200_000)
    return df.sample(n_samples, seed=1, shuffle=True)


# def plot_density_combined(df_peak_dist: pl.DataFrame, out_pdf: Path) -> None:
#    n_tot = len(df_peak_dist)
#    df_peak_dist = subsample_dataframe(df_peak_dist)
#    chart = (
#        (
#            alt.Chart(df_peak_dist)
#            .mark_line()
#            .transform_density(
#                density="dist", as_=["dist", "density"], groupby=["variant"], maxsteps=250, bandwidth=0.01
#            )
#            .encode(x="dist:Q", color="variant:N")
#            .properties(width=500, height=300)
#        )
#        .encode(y=alt.Y("density:Q"))
#        .properties(title="Linear scale")
#    )
#    chart = chart.properties(
#        title=f"Density of target-surrounding peak distances (n_tot = {n_tot} sampled to n = {len(df_peak_dist)})")
#    chart.save(out_pdf)


def plot_density_combined_full(df_peak_dist: pl.DataFrame, out_pdf: Path) -> None:
    variants = df_peak_dist["variant"].unique().to_list()
    collect = []
    for variant in variants:
        df_variant = df_peak_dist.filter(variant=variant)
        dist, density = FFTKDE(bw="ISJ").fit(df_variant["dist"].to_numpy()).evaluate(2 ** 10)
        collect.append(pl.DataFrame({"dist": list(dist), "density": list(density), "variant": variant}))

    dist_min = df_peak_dist["dist"].min()
    dist_max = df_peak_dist["dist"].max()
    df_density = pl.concat(collect).filter((pl.col("dist") >= dist_min) & (pl.col("dist") <= dist_max))

    chart = (
        (
            alt.Chart(df_density)
            .mark_line()
            .encode(x="dist:Q", y="density:Q", color="variant:N")
            .properties(width=500, height=300)
        )
        .properties(title="Linear scale")
    )
    chart = chart.properties(
        title=f"Density of target-surrounding peak distances (N={len(df_peak_dist):,})"
    )
    chart.save(out_pdf)


def plot_density_groups(df_peak_dist: pl.DataFrame, mass_groups: pl.DataFrame, out_peak_density_ranges: Path) -> None:
    # TODO use kdepy as above

    # TODO merge these two functions
    df_peak_dist_grouped = df_peak_dist.sort("mz_target").join_asof(
        mass_groups, left_on="mz_target", right_on="mz_min", strategy="backward"
    )
    df_peak_dist_grouped = subsample_dataframe(df_peak_dist_grouped)
    c = (
        alt.Chart(df_peak_dist_grouped)
        .mark_line()
        .transform_density(
            density="dist", as_=["dist", "density"], groupby=["variant", "mass_group"], maxsteps=250, bandwidth=0.01
        )
        .encode(x="dist:Q", color="variant:N", row="mass_group:N")
        .properties(width=500, height=300)
    )
    c_lin = c.encode(y=alt.Y("density:Q")).properties(title="Linear scale")
    chart = c_lin
    # c_log = c.encode(y=alt.Y("density:Q").scale(type="log", domainMin=1e-4, clamp=True)).properties(title="Log scale")
    # chart = c_lin | c_log
    chart = chart.properties(title="Density of target-surrounding peak distances (grouped by mass range)")
    chart.save(out_peak_density_ranges)


def qc_plot_peak_density(
    table_marker_distances_baseline: Annotated[Path, Option()],
    table_marker_distances_calib: Annotated[Path, Option()],
    output_pdf: Annotated[Path, Option()],
    grouped: Annotated[bool, Option(is_flag=True)],
) -> None:
    vegafusion.enable()

    table_calib = pl.read_parquet(table_marker_distances_calib)
    table_baseline = pl.read_parquet(table_marker_distances_baseline)
    table = pl.concat(
        [
            table_calib.with_columns(variant=pl.lit("calibrated")),
            table_baseline.with_columns(variant=pl.lit("baseline_adj")),
        ]
    )
    if grouped:
        plot_density_groups(
            df_peak_dist=table,
            mass_groups=get_mass_groups(mass_min=table["mz_target"].min(), mass_max=table["mz_target"].max(), n_bins=3),
            out_peak_density_ranges=output_pdf,
        )
    else:
        plot_density_combined_full(df_peak_dist=table, out_pdf=output_pdf)


if __name__ == "__main__":
    typer.run(qc_plot_peak_density)
