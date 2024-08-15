from __future__ import annotations

from pathlib import Path

import cyclopts
import numpy as np
import polars as pl

app = cyclopts.App()


def sample_random_mz(
    min_mz: float, max_mz: float, min_distance_mz: float, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Samples n random mz values in the range [min_mz, max_mz] with a minimum distance of min_distance_mz.
    The samples will include the min and max values (roughly)."""
    mz_range = max_mz - min_mz
    max_distance_mz = mz_range / n
    if min_distance_mz > max_distance_mz:
        raise ValueError("min_distance_mz is too high for the given range and number of samples.")

    distances = rng.uniform(min_distance_mz, max_distance_mz, n)
    mz_offsets = np.cumsum(distances)

    # scale
    mz_offsets *= mz_range / mz_offsets[-1]

    # result
    return mz_offsets + min_mz


@app.default
def generate_synthetic_panel(
    n_labels: int,
    output_path: Path,
    min_mz: float = 800.0,
    max_mz: float = 2400.0,
    min_distance_mz: float = 3.0,
    precision_digits: int = 3,
) -> None:
    """Generates a synthetic panel with n_labels labels and writes it to output_path."""
    rng = np.random.default_rng(0)
    mz_values = sample_random_mz(min_mz, max_mz, min_distance_mz, n_labels, rng)
    mz_values = [round(mz, precision_digits) for mz in mz_values]
    labels = [f"Synthetic_{i:03d}" for i in range(n_labels)]
    panel_df = pl.DataFrame({"label": labels, "mass": mz_values})
    panel_df.write_csv(output_path)


if __name__ == "__main__":
    app()
