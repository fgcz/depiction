from pathlib import Path

import altair as alt
import cyclopts
import polars as pl

from depiction_targeted_preproc.workflow.qc.plot_peak_density import plot_density_combined_full

app = cyclopts.App()


@app.default
def exp_plot_compare_peak_density(
    tables_marker_distances_calib: list[Path],
    table_marker_distance_uncalib: Path,
    output_pdf: Path,
) -> None:
    alt.data_transformers.enable("vegafusion")

    table = pl.concat(
        [
            pl.read_parquet(path).with_columns(variant=pl.lit(path.parents[1].name))
            for path in tables_marker_distances_calib
        ]
    )
    table = pl.concat([table, pl.read_parquet(table_marker_distance_uncalib).with_columns(variant=pl.lit("uncalib"))])

    plot_density_combined_full(df_peak_dist=table, out_pdf=output_pdf)


if __name__ == "__main__":
    app()
