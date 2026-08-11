"""Compares everything in two finished chunk directories except the imzML pairs.

`compare_imzml.py` covers those, separately, because it has to run under both trees' readers.
This one runs in this tree only -- it needs `depiction` to read an OME-TIFF or a
`MultiChannelImage`, and reading each format with its own writer's library is the right way
round here: the question is whether the two pipelines produced the same image, not whether
two libraries agree about a file.

The comparison is by value throughout. An OME-TIFF carries a UUID and a creation date, and a
`.sd.zarr` is a directory of separately-written chunks, so neither is byte-comparable between
two runs of the *same* tree, let alone two trees.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cyclopts
import numpy as np
import spatialdata

from depiction.image.multi_channel_image import MultiChannelImage
from depiction.image.ome_tiff import OmeTiff

app = cyclopts.App()

#: Written by the pipeline as plain text and produced by code that the migration did not
#: touch (polars, pandas, `ParseMetadata`). They are the control: if these differ, the two
#: runs were not given the same input and nothing further is worth reading.
TEXT_ARTIFACTS = (
    "panels/unstandardized_full.csv",
    "panels/full.csv",
    "panels/full_visualize.csv",
    "panels/calibration.csv",
    "raw_metadata.json",
    "config/process_spectra.yml",
    "config/proc_calibrate.yml",
)

#: `calib_data.hdf5` holds one `MultiChannelImage` per group; see
#: `depiction/calibration/apply/calibrate_image.py`.
CALIB_DATA_GROUPS = ("features_raw", "features_processed", "model_coefs")


def compare_arrays(label: str, left: np.ndarray, right: np.ndarray) -> list[str]:
    problems = []
    if left.dtype != right.dtype:
        problems.append(f"{label}: dtype baseline={left.dtype} current={right.dtype}")
    if left.shape != right.shape:
        return [*problems, f"{label}: shape baseline={left.shape} current={right.shape}"]
    if not np.array_equal(left, right, equal_nan=np.issubdtype(left.dtype, np.floating)):
        differing = int(np.sum(left != right))
        largest = float(np.nanmax(np.abs(left.astype(np.float64) - right.astype(np.float64))))
        problems.append(f"{label}: {differing}/{left.size} values differ, max abs diff {largest:.6g}")
    return problems


def compare_images(label: str, left: MultiChannelImage, right: MultiChannelImage) -> list[str]:
    """Values, channel names and foreground -- the three things an image can disagree about."""
    problems = []
    if left.channel_names != right.channel_names:
        problems.append(f"{label}: channel names differ (baseline {len(left.channel_names)} names)")
    if left.sizes != right.sizes:
        problems.append(f"{label}: sizes baseline={left.sizes} current={right.sizes}")
        return problems
    problems += compare_arrays(f"{label} values", left.data_spatial.values, right.data_spatial.values)
    problems += compare_arrays(f"{label} fg_mask", left.fg_mask.values, right.fg_mask.values)
    return problems


def compare_text(baseline_dir: Path, current_dir: Path) -> list[str]:
    problems = []
    for name in TEXT_ARTIFACTS:
        left, right = baseline_dir / name, current_dir / name
        if not left.is_file() or not right.is_file():
            problems.append(f"{name}: missing in baseline={not left.is_file()} current={not right.is_file()}")
        elif left.read_text() != right.read_text():
            problems.append(f"{name}: text differs")
    return problems


def compare_ome_tiff(baseline_dir: Path, current_dir: Path) -> list[str]:
    name = "images_default.ome.tiff"
    problems = compare_images(
        name,
        OmeTiff.read_image(baseline_dir / name, bg_value=0.0),
        OmeTiff.read_image(current_dir / name, bg_value=0.0),
    )
    # `PixelSize` is a frozen dataclass, so this compares by value and stays right when either
    # side is `None` -- which is what a file declaring no raster now produces.
    left = OmeTiff.read(baseline_dir / name).attrs["pixel_size"]
    right = OmeTiff.read(current_dir / name).attrs["pixel_size"]
    if left != right:
        problems.append(f"{name}: pixel size baseline={left} current={right}")
    return problems


def compare_sd_zarr(baseline_dir: Path, current_dir: Path) -> list[str]:
    name = "images_default.sd.zarr"
    left = spatialdata.read_zarr(baseline_dir / name).images["msi"]
    right = spatialdata.read_zarr(current_dir / name).images["msi"]
    problems = compare_arrays(name, np.asarray(left.values), np.asarray(right.values))
    if list(left.coords["c"].values) != list(right.coords["c"].values):
        problems.append(f"{name}: channel coordinates differ")
    return problems


@app.default()
def main(baseline_dir: Path, current_dir: Path) -> None:
    """Compares two finished chunk directories and exits non-zero on any difference."""
    problems = compare_text(baseline_dir, current_dir)
    problems += compare_images(
        "images_default.hdf5",
        MultiChannelImage.read_hdf5(baseline_dir / "images_default.hdf5"),
        MultiChannelImage.read_hdf5(current_dir / "images_default.hdf5"),
    )
    for group in CALIB_DATA_GROUPS:
        problems += compare_images(
            f"calib_data.hdf5[{group}]",
            MultiChannelImage.read_hdf5(baseline_dir / "calib_data.hdf5", group=group),
            MultiChannelImage.read_hdf5(current_dir / "calib_data.hdf5", group=group),
        )
    problems += compare_ome_tiff(baseline_dir, current_dir)
    problems += compare_sd_zarr(baseline_dir, current_dir)

    print(f"{baseline_dir}\n{current_dir}")
    if problems:
        print(f"DIFFERENT -- {len(problems)} problem(s):")
        for problem in problems:
            print(f"  {problem}")
        sys.exit(1)
    print(
        f"IDENTICAL -- {len(TEXT_ARTIFACTS)} text artifacts, images_default.hdf5, "
        f"{len(CALIB_DATA_GROUPS)} calibration groups, the OME-TIFF and the SpatialData zarr"
    )


if __name__ == "__main__":
    app()
