import altair as alt
import cyclopts
import polars as pl
from pathlib import Path

app = cyclopts.App()


@app.default
def qc_plot_spectra_for_marker(
    marker_surrounding_baseline: Path,
    marker_surrounding_calib: Path,
    output_pdf: Path,
) -> None:
    marker_name = "FN1"
    # marker_name = "CHCA peak-2"

    marker_surrounding_baseline_df = pl.read_parquet(marker_surrounding_baseline)
    marker_surrounding_calib_df = pl.read_parquet(marker_surrounding_calib)

    marker_surrounding_df = pl.concat(
        [
            marker_surrounding_baseline_df.with_columns(variant=pl.lit("baseline_adj")),
            marker_surrounding_calib_df.with_columns(variant=pl.lit("calibrated")),
        ]
    )
    plot_df = marker_surrounding_df.filter(label=marker_name)
    print(plot_df)

    [target_mass] = plot_df["mz_target"].unique().to_list()
    target_mass_df = pl.DataFrame({"mz_peak": [target_mass], "label": [f"target {marker_name}"]})
    target_mass_df = pl.concat(
        [target_mass_df.with_columns(variant=pl.lit(variant)) for variant in ["baseline_adj", "calibrated"]]
    )

    def make_density_plot(df):
        c_density = (
            alt.Chart(df)
            .transform_density(density="mz_peak", as_=["m/z", "peak density"], maxsteps=250, bandwidth=0.03)
            .mark_line()
            .encode(x="m/z:Q", y="peak density:Q")
        )
        c_density_line = alt.Chart(target_mass_df).mark_rule(strokeDash=[8, 4], color="red").encode(x="mz_peak:Q")

        return c_density + c_density_line

    c_density_baseline = make_density_plot(plot_df.filter(variant="baseline_adj"))
    c_density_calibrated = make_density_plot(plot_df.filter(variant="calibrated"))

    # def make_spectrum_plot(df, i_spectrum):
    #    df = df.filter(i_spectrum=i_spectrum)
    #    c_peaks = alt.Chart(df).mark_point().encode(x="m/z:Q", y="dist:Q", color="variant:N")
    #    c_target_indicator = alt.Chart(target_mass_df).mark_rule(strokeDash=[8, 4], color="red").encode(x="m/z:Q")
    #    return c_peaks + c_target_indicator

    ## TODO the dataset currently only contains the closest neighbour, so it does not properly work yet
    ## TODO first fix that, before adding the new plot here
    # c_peaks_spec0 = make_spectrum_plot(plot_df, 0)
    # c_peaks_spec1 = make_spectrum_plot(plot_df, 1)
    ##chart = chart & (c_peaks_spec0 | c_peaks_spec1)

    # chart = ((c_density_baseline | c_density_calibrated).resolve_scale(y="shared") & (c_peaks_spec0 | c_peaks_spec1).resolve_scale(y="shared")).resolve_scale(x="shared")
    chart = (c_density_baseline | c_density_calibrated).resolve_scale(y="shared", x="shared")
    # TODO not using the facet functionality, because it's currently a bit buggy for layered charts...
    ## TODO first only make a density plot
    # c_density_line = alt.Chart(plot_df).transform_density(
    #    density="mz_peak", as_=["mz_peak", "density"], maxsteps=250, bandwidth=0.01, groupby=["variant"]
    # ).mark_line().encode(x="mz_peak:Q", y="density:Q")
    # c_target_indicator = alt.Chart(target_mass_df).mark_rule().encode(x="mz_peak:Q")
    ##chart = c_density_line + c_target_indicator
    ##chart = c_density_line
    # chart = c_density_line + c_target_indicator
    ##chart = chart.encode(color="variant:N")
    # chart = chart.facet(column="variant:N")
    ##chart = chart.encode(column="variant:N")
    ##chart = c_density_line

    # TODO
    # chart = chart.properties(width=600, height=300)
    # chart = chart.resolve_scale(y="shared", x="shared")
    chart.save(output_pdf)


if __name__ == "__main__":
    app()
