from collections import defaultdict
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
    collect = defaultdict(list)
    with read_file.get_reader() as reader:
        for i_spectrum in tqdm(range(reader.n_spectra)):
            for mz_target, label in zip(mz_targets, mz_labels):
                peak_mz = reader.get_spectrum_mz(i_spectrum)
                dist = peak_mz - mz_target
                # keep only the distances within max dist
                sel = np.abs(dist) < mz_max_dist
                dist = dist[sel]
                peak_mz = peak_mz[sel]
                if len(dist):
                    # add to result set
                    collect["i_spectrum"].append(i_spectrum)
                    collect["dist"].append(dist)
                    collect["mz_peak"].append(peak_mz)
                    collect["mz_target"].append(mz_target)
                    collect["label"].append(label)
    return pl.DataFrame(collect).explode(["dist", "mz_peak"])


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
