from pathlib import Path
from typing import Annotated

import numpy as np
import polars as pl
import typer
from typer import Option

from depiction_targeted_preproc.pipeline_config.model import SimulateParameters


def simulate_create_mass_list(
    config_path: Annotated[Path, Option()],
    output_mass_list_path: Annotated[Path, Option()],
) -> None:
    # parse the config
    config = SimulateParameters.parse_yaml(config_path)

    # compute the masses
    lambda_avg = 1.0 + 4.95e-4
    n_labels = config.n_labels
    masses = np.linspace(config.target_mass_min, config.target_mass_max, n_labels)
    masses -= masses % lambda_avg

    # create the output mass list
    df_out = pl.DataFrame(
        {"label": [f"synthetic_{i}" for i in range(n_labels)], "mass": masses, "tol": [0.25] * n_labels}
    )
    df_out.write_csv(output_mass_list_path)


if __name__ == "__main__":
    typer.run(simulate_create_mass_list)
