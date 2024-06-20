from pathlib import Path
from typing import Annotated

import typer
import xarray
from matplotlib import pyplot as plt


def exp_plot_map_single_for_poster(
    input_mass_shift_path: Annotated[Path, typer.Option()],
    output_pdf_path: Annotated[Path, typer.Option()],
) -> None:
    # load all the inputs
    shift_map = xarray.open_dataarray(input_mass_shift_path)

    plt.figure(figsize=(10, 10))
    shift_map.isel(c=0).plot(x="x", y="y", ax=plt.gca(), cmap="coolwarm", vmin=-1, vmax=+1, yincrease=False)
    plt.set_aspect("equal")
    variant = input_mass_shift_path.parent.name
    test_mass = shift_map.coords["c"][0]
    plt.set_title(f"{variant} computed shift for test mass {test_mass:.2f}")

    plt.savefig(output_pdf_path, bbox_inches="tight")


if __name__ == "__main__":
    typer.run(exp_plot_map_single_for_poster)
