from pathlib import Path
from typing import Annotated

import altair as alt
import polars as pl
import typer
from loguru import logger
from typer import Option

from depiction.parallel_ops import ReadSpectraParallel, ParallelConfig
from depiction.persistence import ImzmlReadFile, ImzmlReader
from depiction_targeted_preproc.pipeline_config.model import PipelineParameters
from depiction_targeted_preproc.workflow.qc.plot_calibration_map import get_mass_groups


def _get_chunk_counts(reader: ImzmlReader, spectra_ids: list[int], mass_groups: pl.DataFrame) -> pl.DataFrame:
    collect = []
    for i_spectrum in spectra_ids:
        mz_peaks = reader.get_spectrum_mz(i_spectrum)
        collect.extend(
            [
                {"group_index": g_index, "n_peaks": ((mz_peaks > g_mz_min) & (mz_peaks < g_mz_max)).sum()}
                for g_index, g_mz_min, g_mz_max in zip(
                    mass_groups["group_index"], mass_groups["mz_min"], mass_groups["mz_max"]
                )
            ]
        )
    return pl.DataFrame(collect)


def get_peak_counts(read_peaks: ImzmlReadFile, mass_groups: pl.DataFrame, n_jobs: int) -> pl.DataFrame:
    read_parallel = ReadSpectraParallel.from_config(ParallelConfig(n_jobs=n_jobs))
    return read_parallel.map_chunked(
        read_file=read_peaks, operation=_get_chunk_counts, bind_args={"mass_groups": mass_groups}, reduce_fn=pl.concat
    )


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
    logger.info(f"Mass groups: {mass_groups}")

    peak_counts = get_peak_counts(read_peaks=read_peaks, mass_groups=mass_groups, n_jobs=config.n_jobs)
    logger.info("Peak counts: {peak_counts}")

    plot_df = peak_counts.join(mass_groups, on="group_index", how="left")
    n_peaks = plot_df["n_peaks"].sum()

    chart = (
        alt.Chart(plot_df)
        .mark_bar()
        .encode(x=alt.X("n_peaks:Q").bin(maxbins=50), y="count()", color="mass_group:N")
        .properties(title=f"Peak counts per spectrum (N={n_peaks:,})")
    )
    chart.save(output_pdf)


if __name__ == "__main__":
    typer.run(qc_plot_peak_counts_per_spectrum)
