from __future__ import annotations

from pathlib import Path

import pytest

from depiction_io import ImzyReadFile, get_read_file

IMZML = Path("/tmp/does_not_need_to_exist.imzML")


def test_returns_a_read_file() -> None:
    # The path does not exist, which is deliberate: read files are handed to worker
    # processes, so construction must not touch the disk.
    assert isinstance(get_read_file(IMZML), ImzyReadFile)


def test_vendor_format_is_accepted_where_imzy_can_read_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.platform", "linux")
    assert isinstance(get_read_file(Path("/tmp/acquisition.d")), ImzyReadFile)


def test_vendor_format_on_macos_explains_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    # Nothing but imzy can read a Bruker .d, and on macOS imzy cannot either. Saying so
    # here beats failing somewhere inside the reader.
    monkeypatch.setattr("sys.platform", "darwin")
    with pytest.raises(RuntimeError, match="macOS"):
        get_read_file(Path("/tmp/acquisition.d"))


def test_imzml_is_unaffected_by_the_macos_vendor_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.platform", "darwin")
    assert isinstance(get_read_file(IMZML), ImzyReadFile)


def test_suffix_is_matched_case_insensitively(monkeypatch: pytest.MonkeyPatch) -> None:
    # An .IMZML is still an imzML, so it must not be mistaken for a vendor format and
    # rejected on macOS.
    monkeypatch.setattr("sys.platform", "darwin")
    assert isinstance(get_read_file(Path("/tmp/upper.IMZML")), ImzyReadFile)
