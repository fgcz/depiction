"""Concatenates imzML files into one clash-free coordinate system.

`merge_imzml.MergeImzml` copies each input's coordinates verbatim, which is what
`WriteSpectraParallel` needs to stitch its chunks back together, but means two acquisitions that
both start at (1, 1) end up on top of each other. This module shifts each input onto its own
region instead, the way `depiction.image.horizontal_concat` does for images.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from loguru import logger
from tqdm import tqdm

from depiction_io import ImzmlModeEnum

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from numpy.typing import NDArray

    from depiction_io import GenericReadFile
    from depiction_io.types import GenericWriteFile

AXES = ("x", "y")


@dataclass(frozen=True)
class SpatialPlacement:
    """Where one input's coordinates end up in the concatenated output."""

    source_min: tuple[int, int]
    extent: tuple[int, int]
    target_min: tuple[int, int]
    n_spectra: int

    def shift(self, coordinates: NDArray[np.int64]) -> NDArray[np.int64]:
        """Maps input coordinates into the output coordinate system; a z column passes through."""
        shifted = np.array(coordinates)
        shifted[..., :2] += np.asarray(self.target_min) - np.asarray(self.source_min)
        return shifted


def compute_placements(
    coordinate_arrays: Sequence[NDArray[np.int64]],
    *,
    axis: Literal["x", "y"] = "x",
    spacing: int = 0,
) -> list[SpatialPlacement]:
    """Places each input's bounding box after the previous one along `axis`.

    Every input is normalized to its own bounding box first, so its coordinates start at 1 on the
    other axis and at 1 + the running offset on `axis`. imzML coordinates are 1-based.
    :param coordinate_arrays: one (n_spectra, n_dim) coordinate array per input
    :param axis: the axis along which the inputs are placed next to each other
    :param spacing: number of empty pixels to leave between consecutive inputs
    """
    if axis not in AXES:
        raise ValueError(f"Expected axis to be one of {AXES}, got {axis!r}.")
    if spacing < 0:
        raise ValueError(f"Expected a non-negative spacing, got {spacing}.")
    i_axis = AXES.index(axis)

    placements = []
    offset = 0
    for coordinates in coordinate_arrays:
        coordinates_2d = np.asarray(coordinates)[:, :2]
        source_min = coordinates_2d.min(axis=0)
        extent = coordinates_2d.max(axis=0) - source_min + 1
        target_min = np.ones(2, dtype=int)
        target_min[i_axis] += offset
        placements.append(
            SpatialPlacement(
                source_min=_as_tuple(source_min),
                extent=_as_tuple(extent),
                target_min=_as_tuple(target_min),
                n_spectra=coordinates_2d.shape[0],
            )
        )
        offset += int(extent[i_axis]) + spacing
    return placements


def resolve_output_mode(
    input_files: Sequence[GenericReadFile], mode: Literal["continuous", "processed"] | None = None
) -> ImzmlModeEnum:
    """Returns the mode to write the output in, `None` meaning "continuous if that is possible".

    Inputs that are each continuous but not on one shared m/z axis cannot be written as a
    continuous file; deciding that here rather than letting the writer's alignment check fail
    means the tool does not abort halfway through writing.
    """
    if mode is not None:
        return ImzmlModeEnum.from_imzml_str(mode)
    if not all(input_file.imzml_mode == ImzmlModeEnum.CONTINUOUS for input_file in input_files):
        return ImzmlModeEnum.PROCESSED

    mz_arrays = []
    for input_file in input_files:
        with input_file.reader() as reader:
            mz_arrays.append(reader.get_spectrum_mz(0))
    if all(np.array_equal(mz_arrays[0], mz_arr) for mz_arr in mz_arrays[1:]):
        return ImzmlModeEnum.CONTINUOUS
    logger.info("The inputs are continuous but do not share one m/z axis, writing a processed output instead.")
    return ImzmlModeEnum.PROCESSED


def concat_spatial(
    input_files: Sequence[GenericReadFile],
    output_file: GenericWriteFile,
    *,
    axis: Literal["x", "y"] = "x",
    spacing: int = 0,
) -> list[SpatialPlacement]:
    """Writes all spectra of `input_files` to `output_file`, side by side along `axis`.

    Returns the placement of each input, which is what a caller needs to trace an output pixel
    back to the file it came from.
    """
    coordinate_arrays = [input_file.coordinates for input_file in input_files]
    placements = compute_placements(coordinate_arrays, axis=axis, spacing=spacing)
    for input_file, coordinates in zip(input_files, coordinate_arrays):
        # Counted over all dimensions, so a 3D acquisition's stacked x/y is not reported.
        n_duplicates = len(coordinates) - len(np.unique(coordinates, axis=0))
        if n_duplicates:
            logger.warning(
                f"{n_duplicates} of the {len(coordinates)} spectra of {input_file!r} share a pixel with another "
                f"spectrum of the same file; they stay stacked in the output."
            )

    with output_file.writer() as writer:
        for input_file, coordinates, placement in tqdm(
            list(zip(input_files, coordinate_arrays, placements)), desc=" input file", position=0
        ):
            # Shifting the whole array once, rather than `copy_spectra`'s per-spectrum coordinate
            # lookup, is also what lets this run on any reader rather than only on `GenericReader`.
            shifted = placement.shift(coordinates)
            with input_file.reader() as reader:
                for i_spectrum in tqdm(range(reader.n_spectra), desc=" spectrum", position=1):
                    mz_arr, int_arr = reader.get_spectrum(i_spectrum)
                    writer.add_spectrum(mz_arr, int_arr, shifted[i_spectrum])
    return placements


def build_concat_info(
    input_paths: Sequence[Path],
    placements: Sequence[SpatialPlacement],
    *,
    axis: Literal["x", "y"],
    spacing: int,
    imzml_mode: ImzmlModeEnum,
) -> dict:
    """Returns the provenance record of a concatenation, to be written beside the output.

    An output coordinate maps back to its source file as
    ``source = target - target_min + source_min``.
    """
    return {
        "axis": axis,
        "spacing": spacing,
        "imzml_mode": imzml_mode.name,
        "inputs": [
            {
                "path": str(path),
                "n_spectra": placement.n_spectra,
                "source_min": list(placement.source_min),
                "extent": list(placement.extent),
                "target_min": list(placement.target_min),
            }
            for path, placement in zip(input_paths, placements)
        ],
    }


def _as_tuple(values: NDArray[np.int64]) -> tuple[int, int]:
    return int(values[0]), int(values[1])
