import cyclopts
import numpy as np
import polars as pl
from pathlib import Path

from depiction_targeted_preproc.pipeline_config.model import SimulateParameters

app = cyclopts.App()


@app.default
def simulate_create_mass_list(
    config_path: Path,
    output_mass_list_path: Path,
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
    app()
