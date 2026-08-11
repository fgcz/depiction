import json
from pathlib import Path
from typing import Annotated, Literal

import cyclopts
from loguru import logger

from depiction.tools.concat_spatial_imzml import (
    build_concat_info,
    concat_spatial,
    resolve_output_mode,
)
from depiction.tools.coordinate_summary import (
    format_coordinate_summary,
    format_occupancy_map,
    summarize_coordinates,
)
from depiction_io import ImzmlWriteFile, get_read_file

cmd_imzml = cyclopts.App()


@cmd_imzml.command(name="verify")
def cmd_imzml_verify(imzml_path: Path) -> None | int:
    """Verifies if the .ibd file associated with the .imzML file has
    the correct checksum.

    :param imzml_path: Path to the .imzML file.
    """
    read_file = get_read_file(imzml_path)
    if read_file.is_checksum_valid:
        logger.success(f"Checksum for {imzml_path} is valid.")
    else:
        logger.error(f"Checksum for {imzml_path} is invalid.")
        return 1


@cmd_imzml.command(name="coords")
def cmd_imzml_coords(
    imzml_path: Path,
    *,
    as_json: Annotated[bool, cyclopts.Parameter(name="--json")] = False,
    show_map: Annotated[bool, cyclopts.Parameter(name="--map")] = False,
) -> None:
    """Prints information about the spatial coordinates of the spectra in an .imzML file.

    :param imzml_path: Path to the .imzML file.
    :param as_json: Print the information as JSON instead of text; the occupancy map is left out.
    :param show_map: Also print a map of which pixels of the bounding box are occupied.
    """
    read_file = get_read_file(imzml_path)
    summary = summarize_coordinates(read_file.coordinates, pixel_size=read_file.pixel_size)
    if as_json:
        print(json.dumps({"file": str(imzml_path), **summary}, indent=2))
        return
    print(f"file: {imzml_path}")
    print(format_coordinate_summary(summary))
    if show_map:
        print()
        print(format_occupancy_map(read_file.coordinates))


@cmd_imzml.command(name="concat")
def cmd_imzml_concat(
    input_imzml: list[Path],
    *,
    output_imzml: Path,
    axis: Literal["x", "y"] = "x",
    spacing: int = 0,
    mode: Literal["continuous", "processed"] | None = None,
    overwrite: bool = False,
) -> None:
    """Concatenates .imzML files spatially, placing each input's pixels next to the previous
    input's instead of on top of them.

    The placement of every input is written to a .concat.json file beside the output, which is
    what it takes to trace an output pixel back to the file it came from.

    :param input_imzml: Paths of the .imzML files to concatenate, in the order they are placed.
    :param output_imzml: Path of the .imzML file to write.
    :param axis: The axis along which the inputs are placed next to each other.
    :param spacing: Number of empty pixels to leave between consecutive inputs.
    :param mode: Mode to write the output in; by default continuous when the inputs allow it.
    :param overwrite: Overwrite the output file if it already exists.
    """
    read_files = [get_read_file(path) for path in input_imzml]
    pixel_sizes = {read_file.pixel_size for read_file in read_files}
    if len(pixel_sizes) > 1:
        logger.warning(f"The inputs declare different pixel sizes ({pixel_sizes}), the output grid mixes them.")

    imzml_mode = resolve_output_mode(read_files, mode)
    write_file = ImzmlWriteFile(
        output_imzml,
        imzml_mode=imzml_mode,
        write_mode="w" if overwrite else "x",
        # A grid that mixes rasters has no one pixel size, so declare one only when the inputs agree.
        pixel_size=next(iter(pixel_sizes)) if len(pixel_sizes) == 1 else None,
    )
    placements = concat_spatial(read_files, write_file, axis=axis, spacing=spacing)

    info_path = output_imzml.with_suffix(".concat.json")
    info = build_concat_info(input_imzml, placements, axis=axis, spacing=spacing, imzml_mode=imzml_mode)
    info_path.write_text(json.dumps(info, indent=2))
    logger.success(
        f"Wrote {sum(placement.n_spectra for placement in placements)} spectra to {output_imzml} "
        f"({imzml_mode.name}), placement information in {info_path}."
    )
