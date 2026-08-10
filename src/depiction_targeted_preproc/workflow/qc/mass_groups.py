import numpy as np
import polars as pl


def get_mass_groups(mass_min: float, mass_max: float, n_bins: int = 3, add_group_number: bool = True) -> pl.DataFrame:
    edges = np.linspace(mass_min, mass_max, n_bins + 1)
    mz_max = edges[1:]
    mz_max[-1] += 0.1
    mass_groups = pl.DataFrame({"mz_min": edges[:-1], "mz_max": mz_max, "group_index": range(n_bins)})

    if add_group_number:
        mass_groups = mass_groups.with_columns(
            mass_group=pl.format(
                "group_{} = [{}, {})", pl.col("group_index"), pl.col("mz_min").round(3), pl.col("mz_max").round(3)
            )
        )
    else:
        mass_groups = mass_groups.with_columns(
            mass_group=pl.format("[{}, {})", pl.col("mz_min").round(3), pl.col("mz_max").round(3))
        )
    return mass_groups.sort("mz_min")
