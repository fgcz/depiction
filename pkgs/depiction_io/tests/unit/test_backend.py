from __future__ import annotations

from pathlib import Path

import pytest

from depiction_io import ImzmlReadFile, ImzyReadFile, get_read_file
from depiction_io.backend import BACKEND_ENV_VAR

IMZML = Path("/tmp/does_not_need_to_exist.imzML")


def test_default_is_the_imzy_backend() -> None:
    assert isinstance(get_read_file(IMZML), ImzyReadFile)


def test_explicit_backend_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BACKEND_ENV_VAR, "imzy")
    assert isinstance(get_read_file(IMZML, backend="legacy"), ImzmlReadFile)


def test_environment_variable_selects_the_legacy_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    # The escape hatch, for as long as the legacy parser is still here.
    monkeypatch.setenv(BACKEND_ENV_VAR, "legacy")
    assert isinstance(get_read_file(IMZML), ImzmlReadFile)


def test_suffix_is_matched_case_insensitively(monkeypatch: pytest.MonkeyPatch) -> None:
    # An .IMZML is still an imzML, so the backend setting must apply to it rather than it
    # being mistaken for a vendor format.
    monkeypatch.setenv(BACKEND_ENV_VAR, "legacy")
    assert isinstance(get_read_file(Path("/tmp/upper.IMZML")), ImzmlReadFile)


def test_unknown_backend_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BACKEND_ENV_VAR, "pyimzml")
    with pytest.raises(ValueError, match="Unknown backend"):
        get_read_file(IMZML)


def test_vendor_format_ignores_the_backend_setting(monkeypatch: pytest.MonkeyPatch) -> None:
    # Nothing but imzy can read a Bruker .d, so asking for the legacy backend cannot be
    # honoured; on macOS imzy cannot either, and that has to be said plainly.
    monkeypatch.setenv(BACKEND_ENV_VAR, "legacy")
    monkeypatch.setattr("sys.platform", "linux")
    assert isinstance(get_read_file(Path("/tmp/acquisition.d")), ImzyReadFile)


def test_vendor_format_on_macos_explains_itself(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.platform", "darwin")
    with pytest.raises(RuntimeError, match="macOS"):
        get_read_file(Path("/tmp/acquisition.d"))
