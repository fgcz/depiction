from pathlib import Path
from xml.etree import ElementTree

import pytest

from depiction.persistence.imzml.parser.parse_metadata import ParseMetadata
from depiction.persistence.pixel_size import PixelSize


@pytest.fixture()
def xml_path(request) -> Path:
    return Path(__file__).parent / "chunks" / "parse_metadata" / f"{request.param}.xml"


@pytest.fixture()
def etree(xml_path) -> ElementTree.ElementTree:
    return ElementTree.ElementTree(ElementTree.fromstring(xml_path.read_text()))


@pytest.fixture()
def parse_metadata(etree) -> ParseMetadata:
    return ParseMetadata(etree=etree)


@pytest.fixture()
def expected_checksums(xml_path) -> dict[str, str]:
    if xml_path.stem == "checksums_none":
        return {}
    elif xml_path.stem == "checksums_md5":
        return {"md5": "00000000000111111111112222222222"}
    elif xml_path.stem == "checksums_sha1":
        return {"sha1": "abcdef"}
    elif xml_path.stem == "checksums_sha256":
        return {"sha256": "aaaaaa"}
    elif xml_path.stem == "checksums_multiple":
        return {"md5": "00000000000111111111112222222222", "sha256": "aaaaaa"}
    else:
        raise NotImplementedError


@pytest.mark.parametrize(
    "xml_path",
    ["checksums_none", "checksums_md5", "checksums_sha1", "checksums_sha256", "checksums_multiple"],
    indirect=True,
)
def test_ibd_checksums(parse_metadata: ParseMetadata, expected_checksums) -> None:
    assert parse_metadata.ibd_checksums == expected_checksums


@pytest.mark.parametrize("xml_path", ["pixel_size_2d", "pixel_size_none"], indirect=True)
def test_pixel_size(parse_metadata: ParseMetadata, xml_path: Path) -> None:
    if xml_path.stem == "pixel_size_2d":
        assert parse_metadata.pixel_size == PixelSize(size_x=50, size_y=20, unit="micrometer")
    elif xml_path.stem == "pixel_size_none":
        assert parse_metadata.pixel_size is None
    else:
        raise NotImplementedError
