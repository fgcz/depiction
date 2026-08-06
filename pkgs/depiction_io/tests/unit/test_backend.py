from __future__ import annotations

from pathlib import Path

import pytest

from depiction_io import ImzmlReadFile, ImzyReadFile, get_read_file
from depiction_io.backend import BACKEND_ENV_VAR

IMZML = Path("/tmp/does_not_need_to_exist.imzML")


def test_default_is_the_legacy_backend() -> None:
    # The imzy backend is not validated against real acquisitions yet, so a caller that
    # expresses no preference must keep getting the parser that is.
    assert isinstance(get_read_file(IMZML), ImzmlReadFile)


def test_explicit_backend_wins(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BACKEND_ENV_VAR, "legacy")
    assert isinstance(get_read_file(IMZML, backend="imzy"), ImzyReadFile)


def test_environment_variable_selects_imzy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(BACKEND_ENV_VAR, "imzy")
    assert isinstance(get_read_file(IMZML), ImzyReadFile)


def test_suffix_is_matched_case_insensitively() -> None:
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
