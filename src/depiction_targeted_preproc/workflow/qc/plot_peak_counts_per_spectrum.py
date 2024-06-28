from pathlib import Path
from typing import Annotated
import altair as alt
import polars as pl
import typer
from typer import Option

from depiction.persistence import ImzmlReadFile
from depiction_targeted_preproc.pipeline_config.model import PipelineParameters
from depiction_targeted_preproc.workflow.qc.plot_calibration_map import get_mass_groups


def get_peak_counts(read_peaks: ImzmlReadFile, mass_groups: pl.DataFrame, n_jobs: int) -> pl.DataFrame:
    # TODO parallelize, over all !!!FIXME
    collect = []
    with read_peaks.reader() as reader:
        for i_spectrum in range(1000):
            group_index, group_mz_min, group_mz_max = mass_groups.select(["group_index", "mz_min", "mz_max"])
            mz_peaks = reader.get_spectrum_mz(i_spectrum)
            collect.extend(
                [
                    {"group_index": g_index, "n_peaks": ((mz_peaks > g_mz_min) & (mz_peaks < g_mz_max)).sum()}
                    for g_index, g_mz_min, g_mz_max in zip(group_index, group_mz_min, group_mz_max)
                ]
            )
    return pl.DataFrame(collect)


def qc_plot_peak_counts_per_spectrum(
    imzml_peaks: Annotated[Path, Option()],
    output_pdf: Annotated[Path, Option()],
    config_path: Annotated[Path, Option()],
) -> None:
    config = PipelineParameters.parse_yaml(config_path)
    read_peaks = ImzmlReadFile(imzml_peaks)
    with read_peaks.reader() as reader:
        mass_min, mass_max = reader.get_spectrum_mz(0)[[0, -1]]
    mass_groups = get_mass_groups(mass_min=mass_min, mass_max=mass_max, n_bins=5)
    print(mass_groups)

    peak_counts = get_peak_counts(read_peaks=read_peaks, mass_groups=mass_groups, n_jobs=config.n_jobs)
    print(peak_counts)

    plot_df = peak_counts.join(mass_groups, on="group_index", how="left")

    chart = alt.Chart(plot_df).mark_bar().encode(x=alt.X("n_peaks:Q").bin(maxbins=50), y="count()")
    chart = chart | chart.encode(color="mass_group:N")
    chart.save(output_pdf)


if __name__ == "__main__":
    typer.run(qc_plot_peak_counts_per_spectrum)
