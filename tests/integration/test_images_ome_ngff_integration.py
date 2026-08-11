"""The OME-NGFF export, written through to a real .ome.zarr and read back.

Nothing else exercises this path: the `.ome.zarr` artifact is commented out of
`ARTIFACT_FILES_MAPPING`, which is how the module came to sit for a while against a
`bioio.writers.OmeZarrWriter` that no longer exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from bioio import BioImage
from xarray import DataArray

from depiction.image import MultiChannelImage
from depiction_io.imzml.metadata import Metadata
from depiction_io.pixel_size import PixelSize
from depiction_targeted_preproc.workflow.vis.images_ome_ngff import vis_images_ome_ngff

CHANNEL_NAMES = ["mz_100", "mz_200"]


@pytest.fixture
def input_paths(tmp_path: Path) -> tuple[Path, Path]:
    """A two-channel 3x4 image on disk, and the path its metadata should be written to."""
    data = DataArray(
        np.arange(2 * 3 * 4, dtype=float).reshape((2, 3, 4)) + 1.0,
        dims=["c", "y", "x"],
        coords={"c": CHANNEL_NAMES},
    )
    netcdf_path = tmp_path / "images.hdf5"
    MultiChannelImage.from_spatial(data).write_hdf5(netcdf_path)
    return netcdf_path, tmp_path / "raw_metadata.json"


def _write_metadata(path: Path, pixel_size: PixelSize | None) -> None:
    metadata = Metadata(pixel_size=pixel_size, data_processing=[], software=[], ibd_checksums={})
    path.write_text(metadata.model_dump_json())


def _axes(zarr_path: Path) -> list[dict]:
    return json.loads((zarr_path / "zarr.json").read_text())["attributes"]["ome"]["multiscales"][0]["axes"]


def test_declared_pixel_size_reaches_the_zarr(input_paths: tuple[Path, Path], tmp_path: Path) -> None:
    netcdf_path, metadata_path = input_paths
    _write_metadata(metadata_path, PixelSize(size_x=10.0, size_y=20.0, unit="micrometer"))
    output_path = tmp_path / "images.ome.zarr"

    vis_images_ome_ngff(netcdf_path, metadata_path, output_path)

    image = BioImage(output_path)
    assert list(image.channel_names) == CHANNEL_NAMES
    assert (image.physical_pixel_sizes.X, image.physical_pixel_sizes.Y) == (10.0, 20.0)
    assert [axis.get("unit") for axis in _axes(output_path)] == [None, "micrometer", "micrometer"]


def test_unknown_pixel_size_is_written_without_a_unit(input_paths: tuple[Path, Path], tmp_path: Path) -> None:
    """NGFF has no way to omit a scale, so the unit is what separates "unknown" from "1 um".

    A scale of 1 with no unit reads as one pixel; the same scale tagged `micrometer` would be the
    fabricated raster this export used to state.
    """
    netcdf_path, metadata_path = input_paths
    _write_metadata(metadata_path, None)
    output_path = tmp_path / "images.ome.zarr"

    vis_images_ome_ngff(netcdf_path, metadata_path, output_path)

    assert [axis.get("unit") for axis in _axes(output_path)] == [None, None, None]


def test_the_image_keeps_its_shape_and_values(input_paths: tuple[Path, Path], tmp_path: Path) -> None:
    """The channel axis must be declared, or bioio labels it `z` and the image gains a depth."""
    netcdf_path, metadata_path = input_paths
    _write_metadata(metadata_path, None)
    output_path = tmp_path / "images.ome.zarr"

    vis_images_ome_ngff(netcdf_path, metadata_path, output_path)

    image = BioImage(output_path)
    assert [axis["name"] for axis in _axes(output_path)] == ["c", "y", "x"]
    assert image.physical_pixel_sizes.Z is None
    expected = MultiChannelImage.read_hdf5(netcdf_path).data_spatial.transpose("c", "y", "x").values
    np.testing.assert_array_equal(np.asarray(image.data).squeeze(axis=(0, 2)), expected)


if __name__ == "__main__":
    pytest.main()
