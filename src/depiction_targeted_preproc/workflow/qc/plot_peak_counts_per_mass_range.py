from pathlib import Path
from typing import Annotated

import altair as alt
import typer
import polars as pl
from loguru import logger
from typer import Option

from depiction.persistence import ImzmlReadFile
from depiction_targeted_preproc.pipeline_config.model import PipelineParameters
from depiction_targeted_preproc.workflow.qc.plot_calibration_map import get_mass_groups
from depiction_targeted_preproc.workflow.qc.plot_peak_counts import get_peak_counts


def qc_plot_peak_counts_per_mass_range(
    imzml_peaks: Annotated[Path, Option()],
    output_pdf: Annotated[Path, Option()],
    config_path: Annotated[Path, Option()],
    n_groups: Annotated[int, Option()] = 10,
) -> None:
    config = PipelineParameters.parse_yaml(config_path)
    read_peaks = ImzmlReadFile(imzml_peaks)
    with read_peaks.reader() as reader:
        mass_min, mass_max = reader.get_spectrum_mz(0)[[0, -1]]
    mass_groups = get_mass_groups(mass_min=mass_min, mass_max=mass_max, n_bins=n_groups, add_group_number=False)
    logger.info(f"Mass groups: {mass_groups}")

    peak_counts = get_peak_counts(read_peaks=read_peaks, mass_groups=mass_groups, n_jobs=config.n_jobs)
    logger.info(f"Peak counts: {peak_counts}")

    plot_df = (
        peak_counts.join(mass_groups, on="group_index", how="left", coalesce=True)
        .group_by(["mass_group", "group_index"])
        .agg(pl.col("n_peaks").sum())
        .sort("group_index")
    )
    logger.info(f"Plot dataframe: {plot_df}")

    chart = alt.Chart(plot_df).mark_bar().encode(x="n_peaks:Q", y=alt.Y("mass_group:N", sort=None))
    chart.save(output_pdf)


if __name__ == "__main__":
    typer.run(qc_plot_peak_counts_per_mass_range)
