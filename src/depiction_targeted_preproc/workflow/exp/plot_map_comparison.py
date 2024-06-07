from pathlib import Path
from typing import Annotated
import xarray
import typer
from matplotlib import pyplot as plt


def exp_plot_map_comparison(
    input_mass_shift_paths: Annotated[list[Path], typer.Argument()],
    output_pdf_path: Annotated[Path, typer.Option()],
) -> None:
    # load all the inputs
    mass_shifts = [xarray.open_dataarray(path) for path in input_mass_shift_paths]

    fig, axs = plt.subplots(1, len(mass_shifts), figsize=(10 * len(mass_shifts), 10))
    for i, shift_map in enumerate(mass_shifts):
        shift_map.isel(c=0).plot(x="x", y="y", ax=axs[i], cmap="coolwarm", vmin=-1, vmax=+1)
        axs[i].set_aspect("equal")
        variant = input_mass_shift_paths[i].parent.name
        test_mass = shift_map.coords["c"][0]
        axs[i].set_title(f"{variant} computed shift for test mass {test_mass:.2f}")

    plt.savefig(output_pdf_path, bbox_inches="tight")

if __name__ == "__main__":
    typer.run(exp_plot_map_comparison)
