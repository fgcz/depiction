from pathlib import Path

import altair as alt
import cyclopts
import polars as pl
from KDEpy import FFTKDE

from depiction_targeted_preproc.workflow.qc.mass_groups import get_mass_groups


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
    # `sorted`, because `Series.unique()` does not preserve order in polars and returned the two
    # variants either way round between runs. That decided the order of `collect`, and so the row
    # order of the concatenated frame, and so which colour each variant was assigned.
    variants = sorted(df_peak_dist["variant"].unique().to_list())
    collect = []
    for variant in variants:
        df_variant = df_peak_dist.filter(variant=variant)
        dist, density = FFTKDE(bw="ISJ").fit(df_variant["dist"].to_numpy()).evaluate(2**10)
        collect.append(pl.DataFrame({"dist": list(dist), "density": list(density), "variant": variant}))

    dist_min = df_peak_dist["dist"].min()
    dist_max = df_peak_dist["dist"].max()
    df_density = (
        pl.concat(collect).filter((pl.col("dist") >= dist_min) & (pl.col("dist") <= dist_max)).sort(["variant", "dist"])
    )

    chart = (
        alt.Chart(df_density)
        .mark_line()
        .encode(x="dist:Q", y="density:Q", color="variant:N")
        .properties(width=500, height=300)
    ).properties(title="Linear scale")
    chart = chart.properties(title=f"Density of target-surrounding peak distances (N={len(df_peak_dist):,})")
    chart.save(out_pdf)


#: Matches the `bandwidth=` and `maxsteps=` that Vega-Lite's `transform_density` was given before
#: the KDE moved in here, so the curve is the same one it always drew. 256 rather than 250 because
#: `FFTKDE` bins onto a grid and a power of two is what that is cheapest on.
_DENSITY_BANDWIDTH = 0.01
_DENSITY_STEPS = 256


def density_by_group(df: pl.DataFrame) -> pl.DataFrame:
    """Evaluates the KDE per (variant, mass_group), returning one row per evaluation step.

    Done here rather than by Vega-Lite's `transform_density` -- which is what
    `plot_density_combined_full` above already does, and what this function's TODO asked for. The
    reason is reproducibility. `transform_density` leaves the reduction to the renderer, so all
    200_000 sampled rows are inlined into the chart spec, and the order they come back in differs
    on every run; the rendered curve differed with it, which is what made this plot useless as a
    regression check. Reducing here means the chart carries a few hundred rows in an order this
    function fixes, and the figure is a function of the data alone.
    """
    collect = []
    for (variant, mass_group), group in df.group_by(["variant", "mass_group"]):
        dist, density = FFTKDE(bw=_DENSITY_BANDWIDTH).fit(group["dist"].to_numpy()).evaluate(_DENSITY_STEPS)
        collect.append(pl.DataFrame({"dist": dist, "density": density, "variant": variant, "mass_group": mass_group}))
    return pl.concat(collect).sort(["mass_group", "variant", "dist"])


def plot_density_groups(df_peak_dist: pl.DataFrame, mass_groups: pl.DataFrame, out_peak_density_ranges: Path) -> None:
    # TODO merge these two functions
    df_peak_dist_grouped = df_peak_dist.sort("mz_target").join_asof(
        mass_groups, left_on="mz_target", right_on="mz_min", strategy="backward"
    )
    df_peak_dist_grouped = subsample_dataframe(df_peak_dist_grouped)

    df_density = density_by_group(df_peak_dist_grouped)

    chart = (
        alt.Chart(df_density)
        .mark_line()
        .encode(x="dist:Q", y=alt.Y("density:Q"), color="variant:N", row="mass_group:N")
        .properties(
            width=500,
            height=300,
            title="Density of target-surrounding peak distances (grouped by mass range)",
        )
    )
    chart.save(out_peak_density_ranges)


app = cyclopts.App()


@app.default
def qc_plot_peak_density(
    table_marker_distances_baseline: Path,
    table_marker_distances_calib: Path,
    output_pdf: Path,
    grouped: bool = False,
) -> None:
    alt.data_transformers.enable("vegafusion")

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
    app()
