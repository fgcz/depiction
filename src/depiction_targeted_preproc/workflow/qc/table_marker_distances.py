from pathlib import Path
from typing import Sequence, Annotated

import numpy as np
import polars as pl
import typer
from tqdm import tqdm

from depiction.persistence import ImzmlReadFile


# TODO maybe rename to marker_surroundings


def get_peak_dist(
    read_file: ImzmlReadFile, mz_targets: Sequence[float], mz_labels: Sequence[str], mz_max_dist: float
) -> pl.DataFrame:
    collect = []
    with read_file.get_reader() as reader:
        for i_spectrum in tqdm(range(reader.n_spectra)):
            spectrum_mz = reader.get_spectrum_mz(i_spectrum)
            for mz_target, label in zip(mz_targets, mz_labels):
                dist = spectrum_mz - mz_target
                # keep only the distances within max dist
                sel = np.abs(dist) < mz_max_dist
                dist = dist[sel]
                marker_mz = spectrum_mz[sel]
                if len(dist):
                    collect.append({
                        "i_spectrum": i_spectrum,
                        "dist": list(dist),
                        "mz_peak": list(marker_mz),
                        "mz_target": mz_target,
                        "label": label,
                    })
    return pl.from_dicts(collect).explode(["dist", "mz_peak"])


def qc_table_marker_distances(
    imzml_peaks: Annotated[Path, typer.Option()],
    mass_list: Annotated[Path, typer.Option()],
    output_table: Annotated[Path, typer.Option()],
    mz_max_dist: Annotated[float, typer.Option()] = 3.0,
) -> None:
    read_file = ImzmlReadFile(imzml_peaks)
    mass_list_df = pl.read_csv(mass_list)
    dist_df = get_peak_dist(
        read_file=read_file,
        mz_targets=mass_list_df["mass"].to_list(),
        mz_labels=mass_list_df["label"].to_list(),
        mz_max_dist=mz_max_dist,
    )
    dist_df.write_parquet(output_table)


def main() -> None:
    """Parses CLI args and calls `qc_table_marker_distances`."""
    typer.run(qc_table_marker_distances)


if __name__ == "__main__":
    main()
