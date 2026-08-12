import altair as alt
import cyclopts
import numpy as np
import polars as pl
from pathlib import Path


def sorted_label_order(df: pl.DataFrame, cutoff: float) -> list[str]:
    """The y-axis category order: markers by detection fraction at the tightest cutoff, best first.

    The second sort key is what makes this reproducible. `fraction` alone is not a total order --
    markers tie routinely, and at the tightest cutoff a great many tie at zero -- and the frame
    reaching this function comes out of `group_by`, which hands its rows back in a different order
    on every run. The tie-break, and with it the entire category order of the plot, was therefore
    effectively random: two runs on byte-identical input produced visibly different figures, which
    is what made this QC output useless as a regression check. `label` is unique within the
    filtered frame, so adding it as a tie-break makes the order total.
    """
    return (
        df.filter(detection_dist=cutoff, variant="calibrated")
        .sort(["fraction", "label"], descending=[True, False])["label"]
        .to_list()
    )


def plot_marker_presence(df_peak_dist: pl.DataFrame, n_spectra: int, out_path: Path, layout_vertical: bool) -> None:
    # Add a `max_dist` column to the dataframe, that indicates the first bin a particular item falls into
    df = df_peak_dist.with_columns(abs_dist=pl.col("dist").abs()).sort("abs_dist")
    df_cutoffs = pl.DataFrame({"max_dist": [0.005, 0.05, 0.1, 0.2, 0.3, 0.4, np.inf]}).sort("max_dist")
    df = df.join_asof(df_cutoffs, left_on="abs_dist", right_on="max_dist", strategy="forward").filter(
        pl.col("max_dist").is_finite()
    )

    # For every (label, variant, i_spectrum) compute the minimal `max_dist` value, i.e. the earliest threshold
    df = df.group_by(["label", "variant", "i_spectrum"]).agg(detection_dist=pl.min("max_dist"))

    # Aggregate into a fraction
    df = df.group_by(["label", "variant", "detection_dist"]).agg(fraction=pl.n_unique("i_spectrum") / n_spectra)
    print(df)

    # sort labels by the calibrated image's detection_dist
    sorted_labels = sorted_label_order(df, cutoff=df_cutoffs["max_dist"][0])

    # Same reason as in `sorted_label_order`, for the frame itself: the segments of each bar are
    # stacked in the order the rows arrive, and `group_by` does not fix that order.
    df = df.sort(["label", "variant", "detection_dist"])

    layout_config = (
        {"column": alt.Column("variant:N", title=None)}
        if not layout_vertical
        else {"row": alt.Row("variant:N", title=None, header=alt.Header(orient="top"))}
    )
    size_config = {"width": 600} if not layout_vertical else {"width": 500, "height": 300}
    c = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("sum(fraction):Q", scale=alt.Scale(domain=[0, 1]), title="Fraction of spectra with peaks detected"),
            y=alt.Y("label:N", sort=sorted_labels, title=None),
            color=alt.Color("detection_dist:N", legend=alt.Legend(title="Max distance cutoff", orient="top")),
            **layout_config,
        )
        .properties(title=alt.Title("Marker presence at different mass windows", anchor="middle"), **size_config)
        .configure_axis(labelFontSize=14, titleFontSize=16)
        .configure_header(titleFontSize=16, labelFontSize=16, labelFontWeight="bold")
        .configure_title(fontSize=20)
        .configure_legend(labelFontSize=14, titleFontSize=16)
    )
    c.save(out_path)


app = cyclopts.App()


@app.default
def qc_plot_marker_presence(
    table_marker_distances_baseline: Path,
    table_marker_distances_calib: Path,
    output_pdf: Path,
    layout_vertical: bool = False,
) -> None:
    table_calib = pl.read_parquet(table_marker_distances_calib)
    table_baseline = pl.read_parquet(table_marker_distances_baseline)
    table = pl.concat(
        [
            table_calib.with_columns(variant=pl.lit("calibrated")),
            table_baseline.with_columns(variant=pl.lit("baseline_adj")),
        ]
    )
    plot_marker_presence(
        df_peak_dist=table,
        n_spectra=table["i_spectrum"].n_unique(),
        out_path=output_pdf,
        layout_vertical=layout_vertical,
    )


if __name__ == "__main__":
    app()
