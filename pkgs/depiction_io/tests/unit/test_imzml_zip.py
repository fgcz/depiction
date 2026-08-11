import zipfile
from pathlib import Path

import pytest

from depiction_io.imzml_zip import ImzmlZip

IMZML_BYTES = b"<mzML/>"
IBD_BYTES = b"\x00\x01\x02\x03"


def _write_zip(path: Path, prefix: str = "") -> Path:
    with zipfile.ZipFile(path, "w") as file:
        file.writestr(f"{prefix}sample.imzML", IMZML_BYTES)
        file.writestr(f"{prefix}sample.ibd", IBD_BYTES)
    return path


@pytest.fixture()
def zip_at_root(tmp_path) -> Path:
    return _write_zip(tmp_path / "root.zip")


@pytest.fixture()
def zip_in_subdirectory(tmp_path) -> Path:
    return _write_zip(tmp_path / "sub.zip", prefix="acquisition/")


@pytest.mark.parametrize("fixture_name", ["zip_at_root", "zip_in_subdirectory"])
def test_extract_returns_an_existing_path(request, fixture_name, tmp_path, monkeypatch) -> None:
    """Both layouts the class docstring claims to support, with a relative output directory.

    The relative directory is the case that used to produce `out/out/sample.imzML` while
    returning `out/sample.imzML`, so the returned path did not exist at all.
    """
    monkeypatch.chdir(tmp_path)

    result = ImzmlZip(request.getfixturevalue(fixture_name)).extract("out")

    assert result == Path("out/sample.imzML")
    assert result.read_bytes() == IMZML_BYTES
    assert result.with_suffix(".ibd").read_bytes() == IBD_BYTES


def test_extract_when_filename_given(zip_in_subdirectory, tmp_path) -> None:
    result = ImzmlZip(zip_in_subdirectory).extract(tmp_path / "out", imzml_filename="renamed.imzML")

    assert result == tmp_path / "out" / "renamed.imzML"
    assert result.read_bytes() == IMZML_BYTES
    assert (tmp_path / "out" / "renamed.ibd").read_bytes() == IBD_BYTES


def test_extract_when_archive_invalid(tmp_path) -> None:
    path = tmp_path / "empty.zip"
    with zipfile.ZipFile(path, "w") as file:
        file.writestr("readme.txt", "no imzML here")

    with pytest.raises(ValueError, match="Expected exactly one"):
        ImzmlZip(path).extract(tmp_path / "out")


if __name__ == "__main__":
    pytest.main()
