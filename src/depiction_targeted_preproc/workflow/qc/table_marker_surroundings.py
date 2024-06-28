from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer

from depiction.parallel_ops import ParallelConfig, ReadSpectraParallel
from depiction.persistence import ImzmlReadFile, ImzmlReader
from depiction_targeted_preproc.pipeline_config.model import PipelineParameters


def _get_marker_surroundings_chunk(
    reader: ImzmlReader, spectra_ids: list[int], targets: list[tuple[float, str]], mz_max_dist: float
) -> pl.DataFrame:
    collect = []
    for i_spectrum in spectra_ids:
        spectrum_mz = reader.get_spectrum_mz(i_spectrum)
        for mz_target, label in targets:
            dist = spectrum_mz - mz_target
            # keep only the distances within max dist
            sel = np.abs(dist) < mz_max_dist
            dist = dist[sel]
            marker_mz = spectrum_mz[sel]
            if len(dist):
                collect.append(
                    {
                        "i_spectrum": i_spectrum,
                        "dist": list(dist),
                        "mz_peak": list(marker_mz),
                        "mz_target": mz_target,
                        "label": label,
                    }
                )
    return pl.from_dicts(collect).explode(["dist", "mz_peak"])


def get_marker_surroundings(
    read_file: ImzmlReadFile, mz_targets: Sequence[float], mz_labels: Sequence[str], mz_max_dist: float, n_jobs: int
) -> pl.DataFrame:
    read_parallel = ReadSpectraParallel.from_config(ParallelConfig(n_jobs=n_jobs))
    return read_parallel.map_chunked(
        read_file=read_file,
        operation=_get_marker_surroundings_chunk,
        bind_args={
            "targets": [(mz_target, label) for mz_target, label in zip(mz_targets, mz_labels)],
            "mz_max_dist": mz_max_dist,
        },
        reduce_fn=pl.concat,
    )


def qc_table_marker_surroundings(
    imzml_peaks: Annotated[Path, typer.Option()],
    mass_list: Annotated[Path, typer.Option()],
    config_path: Annotated[Path, typer.Option()],
    output_table: Annotated[Path, typer.Option()],
    mz_max_dist: Annotated[float, typer.Option()] = 3.0,
) -> None:
    config = PipelineParameters.parse_yaml(config_path)
    n_jobs = config.n_jobs

    read_file = ImzmlReadFile(imzml_peaks)
    mass_list_df = pl.read_csv(mass_list)
    dist_df = get_marker_surroundings(
        read_file=read_file,
        mz_targets=mass_list_df["mass"].to_list(),
        mz_labels=mass_list_df["label"].to_list(),
        mz_max_dist=mz_max_dist,
        n_jobs=n_jobs,
    )
    dist_df.write_parquet(output_table)


def main() -> None:
    """Parses CLI args and calls `qc_table_marker_distances`."""
    typer.run(qc_table_marker_surroundings)


if __name__ == "__main__":
    main()
