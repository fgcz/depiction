"""What `ImzmlWriteFile` leaves on disk, written through to real files.

The unit tests next door patch `Path.exists`, `Path.unlink` and `ImzmlWriter.open`, so they
pin the calls the write file makes rather than their result. What a caller actually depends on
is narrower and not visible from there: once the `with` block is over, is there a file at the
output path, and if there was one before, is it still the one that is there.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pytest_mock import MockerFixture

from depiction_io import ImzmlModeEnum, ImzmlWriteFile, ImzyReadFile

MZ_ARR = np.array([100.0, 200.0, 300.0])
INT_ARR = np.array([1.0, 2.0, 3.0])


def _write(path: Path, n_spectra: int, *, write_mode: str = "x") -> None:
    write_file = ImzmlWriteFile(path, imzml_mode=ImzmlModeEnum.PROCESSED, write_mode=write_mode)
    with write_file.writer() as writer:
        for i in range(n_spectra):
            writer.add_spectrum(MZ_ARR, INT_ARR, (i + 1, 1))


def _contents(directory: Path) -> dict[str, bytes]:
    """Every file in `directory`, including the dot-prefixed temporaries imzy writes.

    Comparing the whole mapping rather than probing individual paths is what makes a leftover
    temporary, or a file that changed without its name changing, fail the assertion too.
    """
    return {path.name: path.read_bytes() for path in sorted(directory.iterdir())}


def test_nothing_written_leaves_no_file(tmp_path: Path) -> None:
    write_file = ImzmlWriteFile(tmp_path / "out.imzML", imzml_mode=ImzmlModeEnum.PROCESSED)
    with pytest.raises(ValueError, match="without any spectra"), write_file.writer():
        pass
    assert _contents(tmp_path) == {}


def test_nothing_written_over_an_existing_file_leaves_no_file(tmp_path: Path) -> None:
    # "w" truncates on open, so the old file is gone either way; what matters is that nothing
    # takes its place. `imzml concat --overwrite` and a `SubsampleImzml` ratio that rounds down
    # to zero spectra both reach this.
    path = tmp_path / "out.imzML"
    _write(path, n_spectra=2)

    write_file = ImzmlWriteFile(path, imzml_mode=ImzmlModeEnum.PROCESSED, write_mode="w")
    with pytest.raises(ValueError, match="without any spectra"), write_file.writer():
        pass
    assert _contents(tmp_path) == {}


def test_failure_in_the_body_leaves_no_file(tmp_path: Path) -> None:
    # `close()` renames the partial output into place, so taking that door on the way out of a
    # failed run leaves a truncated file that is indistinguishable from a complete one.
    write_file = ImzmlWriteFile(tmp_path / "out.imzML", imzml_mode=ImzmlModeEnum.PROCESSED)
    with pytest.raises(ZeroDivisionError), write_file.writer() as writer:
        writer.add_spectrum(MZ_ARR, INT_ARR, (1, 1))
        1 / 0
    assert _contents(tmp_path) == {}


def test_failure_in_the_body_over_an_existing_file_leaves_no_file(tmp_path: Path) -> None:
    # The dangerous outcome is not the old file going away -- "w" promises that -- but a
    # truncated new one landing in its place, which parses and is quietly short of spectra.
    path = tmp_path / "out.imzML"
    _write(path, n_spectra=2)

    write_file = ImzmlWriteFile(path, imzml_mode=ImzmlModeEnum.PROCESSED, write_mode="w")
    with pytest.raises(ZeroDivisionError), write_file.writer() as writer:
        writer.add_spectrum(MZ_ARR, INT_ARR, (1, 1))
        1 / 0
    assert _contents(tmp_path) == {}


def test_interrupt_while_closing_leaves_no_file(tmp_path: Path, mocker: MockerFixture) -> None:
    # imzy tidies its temporaries away when finalising raises an `Exception`, but a Ctrl-C is
    # not one, so the close has to happen somewhere the discard can still catch it.
    write_file = ImzmlWriteFile(tmp_path / "out.imzML", imzml_mode=ImzmlModeEnum.PROCESSED)
    with pytest.raises(KeyboardInterrupt), write_file.writer() as writer:
        writer.add_spectrum(MZ_ARR, INT_ARR, (1, 1))
        mocker.patch.object(writer._imzml_writer, "_write_xml", side_effect=KeyboardInterrupt)
    assert _contents(tmp_path) == {}


def test_overwriting_replaces_the_previous_file(tmp_path: Path) -> None:
    # The counterpart to the two above: refusing to leave a partial file must not cost the
    # ability to write a whole one. Checked before the reader is built, since reading writes an
    # `.icache`.
    path = tmp_path / "out.imzML"
    _write(path, n_spectra=2)
    _write(path, n_spectra=5, write_mode="w")

    assert set(_contents(tmp_path)) == {"out.imzML", "out.ibd"}
    assert ImzyReadFile(path).n_spectra == 5


def test_overwriting_when_the_ibd_is_missing(tmp_path: Path) -> None:
    # The `.ibd` was unlinked without `missing_ok`, so an output whose `.ibd` had gone missing
    # raised `FileNotFoundError` -- after the `.imzML` had already been removed.
    path = tmp_path / "out.imzML"
    _write(path, n_spectra=2)
    path.with_suffix(".ibd").unlink()

    _write(path, n_spectra=3, write_mode="w")
    assert ImzyReadFile(path).n_spectra == 3


def test_write_mode_x_refuses_an_existing_file(tmp_path: Path) -> None:
    path = tmp_path / "out.imzML"
    _write(path, n_spectra=2)
    before = _contents(tmp_path)

    write_file = ImzmlWriteFile(path, imzml_mode=ImzmlModeEnum.PROCESSED)
    with pytest.raises(ValueError, match="already exists"), write_file.writer():
        pass
    assert _contents(tmp_path) == before


def test_invalid_write_mode_creates_nothing(tmp_path: Path) -> None:
    write_file = ImzmlWriteFile(tmp_path / "out.imzML", imzml_mode=ImzmlModeEnum.PROCESSED, write_mode="a")
    with pytest.raises(ValueError, match="Invalid write mode"), write_file.writer():
        pass
    assert _contents(tmp_path) == {}
