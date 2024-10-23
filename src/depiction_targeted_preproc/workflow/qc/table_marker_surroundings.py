import cyclopts
import numpy as np
import polars as pl
from collections.abc import Sequence
from pathlib import Path

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


app = cyclopts.App()


@app.default
def qc_table_marker_surroundings(
    imzml_peaks: Path,
    mass_list: Path,
    config_path: Path,
    output_table: Path,
    mz_max_dist: float = 3.0,
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


if __name__ == "__main__":
    app()
