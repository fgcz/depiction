from __future__ import annotations

import pytest

from depiction_io.imzml.metadata import Metadata
from depiction_io.pixel_size import PixelSize


def test_pixel_size_is_optional() -> None:
    # It was required until a file declaring no `IMS:1000046` turned every parse of it into a
    # `ValidationError`, and the one caller that handled that substituted a dummy 1 um.
    metadata = Metadata(data_processing=[], software=[], ibd_checksums={})
    assert metadata.pixel_size is None


@pytest.mark.parametrize("pixel_size", [None, PixelSize(size_x=50.0, size_y=20.0, unit="micrometer")])
def test_json_round_trip(pixel_size: PixelSize | None) -> None:
    # The pipeline passes this model between steps as JSON on disk, so an absent pixel size has to
    # survive serialization as an absent one rather than as a default.
    metadata = Metadata(pixel_size=pixel_size, data_processing=["a"], software=["b"], ibd_checksums={"md5": "0"})
    assert Metadata.model_validate_json(metadata.model_dump_json()) == metadata


if __name__ == "__main__":
    pytest.main()
