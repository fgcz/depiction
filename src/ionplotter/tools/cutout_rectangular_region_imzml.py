import argparse
import os
import json

import numpy as np

from ionplotter.persistence import ImzmlReadFile, ImzmlWriteFile


class CutoutRectangularRegionImzml:
    """Cuts out a rectangular region from an imzML file. Ranges are inclusive."""

    def __init__(
        self,
        *,
        read_file: ImzmlReadFile,
        x_range_abs: tuple[int, int],
        y_range_abs: tuple[int, int],
        verbose: bool,
    ) -> None:
        self._read_file = read_file
        self._x_range_abs = x_range_abs
        self._y_range_abs = y_range_abs
        self._verbose = verbose

    @classmethod
    def from_relative_ranges(
        cls,
        read_file: ImzmlReadFile,
        x_range_rel: tuple[float, float],
        y_range_rel: tuple[float, float],
        verbose: bool = True,
    ) -> "CutoutRectangularRegionImzml":
        """Constructs a new instance from relative ranges, i.e. from 0 to 1 for each dimension.
        :param read_file: the read file
        :param x_range_rel: the relative x range
        :param y_range_rel: the relative y range
        :param verbose: whether to print verbose output
        """
        x_range_abs, y_range_abs = cls._convert_relative_to_absolute(
            read_file=read_file,
            x_range_rel=x_range_rel,
            y_range_rel=y_range_rel,
        )
        return cls.from_absolute_ranges(
            read_file=read_file, x_range_abs=x_range_abs, y_range_abs=y_range_abs, verbose=verbose
        )

    @classmethod
    def from_absolute_ranges(
        cls,
        read_file: ImzmlReadFile,
        x_range_abs: tuple[int, int],
        y_range_abs: tuple[int, int],
        verbose: bool = True,
    ) -> "CutoutRectangularRegionImzml":
        """Constructs a new instance from absolute ranges, i.e. in the coordinate system of the input file.
        :param read_file: the read file
        :param x_range_abs: the absolute x range
        :param y_range_abs: the absolute y range
        :param verbose: whether to print verbose output
        """
        return cls(
            read_file=read_file,
            x_range_abs=x_range_abs,
            y_range_abs=y_range_abs,
            verbose=verbose,
        )

    @classmethod
    def _convert_relative_to_absolute(
        cls,
        read_file: ImzmlReadFile,
        x_range_rel: tuple[float, float],
        y_range_rel: tuple[float, float],
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        coordinates = read_file.coordinates_2d
        x_min = coordinates[:, 0].min()
        x_max = coordinates[:, 0].max()
        y_min = coordinates[:, 1].min()
        y_max = coordinates[:, 1].max()
        x_extent = x_max - x_min + 1
        y_extent = y_max - y_min + 1
        x_range_abs = (
            max(round(x_min + x_range_rel[0] * x_extent), x_min),
            min(round(x_min + x_range_rel[1] * x_extent), x_max),
        )
        y_range_abs = (
            max(round(y_min + y_range_rel[0] * y_extent), y_min),
            min(round(y_min + y_range_rel[1] * y_extent), y_max),
        )
        return x_range_abs, y_range_abs

    def write_imzml(self, write_file: ImzmlWriteFile) -> None:
        """Writes the result to the specified file."""
        if self._verbose:
            print(f"Copying range: x = {self._x_range_abs}, y = {self._y_range_abs}")

        with self._read_file.reader() as reader, write_file.writer() as writer:
            # determine the indices of the spectra to copy over
            x_min, x_max = self._x_range_abs
            y_min, y_max = self._y_range_abs
            coordinates = reader.coordinates_2d
            spectra_indices = np.where(
                (x_min <= coordinates[:, 0])
                & (coordinates[:, 0] <= x_max)
                & (y_min <= coordinates[:, 1])
                & (coordinates[:, 1] <= y_max)
            )[0]
            if self._verbose:
                print(f"Copying {len(spectra_indices)} spectra.")
            if len(spectra_indices) == 0:
                print("No spectra to copy. Exiting.")
                return

            # copy the relevant spectra
            writer.copy_spectra(reader=reader, spectra_indices=spectra_indices)

    def write_operation_info(self) -> None:
        """Writes information about the operation performed into an additional JSON file."""
        information = {
            "input_imzml": self._read_file.imzml_file,
            "x_range_abs": self._x_range_abs,
            "y_range_abs": self._y_range_abs,
        }
        output_path = self._read_file.imzml_file.with_suffix(".cutout_rectangular_region.json")
        with output_path.open("w") as f:
            json.dump(information, f, indent=1)


def main_cutout_rectangular_region_imzml(
    input_imzml: str,
    output_imzml: str,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    relative: bool,
) -> None:
    """Cuts out a rectangular region from an imzML file.
    :param input_imzml: the input imzML file
    :param output_imzml: the output imzML file
    :param xmin: the minimum x value
    :param xmax: the maximum x value
    :param ymin: the minimum y value
    :param ymax: the maximum y value
    :param relative: whether the provided values are relative (0 to 1) or absolute
    """
    x_range = (xmin, xmax)
    y_range = (ymin, ymax)

    read_file = ImzmlReadFile(input_imzml)
    write_file = ImzmlWriteFile(output_imzml, imzml_mode=read_file.imzml_mode)

    if relative:
        cutout = CutoutRectangularRegionImzml.from_relative_ranges(
            read_file=read_file,
            x_range_rel=x_range,
            y_range_rel=y_range,
        )
    else:
        cutout = CutoutRectangularRegionImzml.from_absolute_ranges(
            read_file=read_file,
            x_range_abs=x_range,
            y_range_abs=y_range,
        )

    cutout.write_imzml(write_file=write_file)
    cutout.write_operation_info()


def main() -> None:
    """Provides CLI for main_cutout_rectangular_region_imzml."""
    parser = argparse.ArgumentParser()
    parser.add_argument("input_imzml", type=str, help="The input imzML file.")
    parser.add_argument("output_imzml", type=str, help="The output imzML file.")
    parser.add_argument("--xmin", type=float, help="The minimum x value.")
    parser.add_argument("--xmax", type=float, help="The maximum x value.")
    parser.add_argument("--ymin", type=float, help="The minimum y value.")
    parser.add_argument("--ymax", type=float, help="The maximum y value.")
    parser.add_argument(
        "--relative",
        action="store_true",
        help=(
            "If set, the provided values are relative (0 to 1), otherwise they are absolute in the coordinate system "
            "of the input file."
        ),
    )
    args = vars(parser.parse_args())
    main_cutout_rectangular_region_imzml(**args)


if __name__ == "__main__":
    main()
