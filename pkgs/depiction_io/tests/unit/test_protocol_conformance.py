"""Every concrete backend against the protocol it claims to implement.

`RamReader` and `RamReadFile` declared no base for a long time, so they inherited none of the
default implementations in `types.py` and were missing nine members between them. Passing one
where a `GenericReader`/`GenericReadFile` was expected raised `AttributeError` on the first
protocol method the caller touched -- including `GenerateIonImage.generate_ion_images_for_file`,
which reaches for `coordinates_array_2d`. Nothing failed, because nothing checked.

The parametrized test below is that check, and it covers the imzy and imzML backends for free.
The behavioural tests after it are the other half: a member can be *present* and still be the
protocol's own `raise NotImplementedError` stub, which a membership check cannot see.
"""

from __future__ import annotations

import numpy as np
import pytest

from depiction_io import ImzmlModeEnum
from depiction_io.imzml.imzml_write_file import ImzmlWriteFile
from depiction_io.imzml.imzml_writer import ImzmlWriter
from depiction_io.imzy_backend.imzy_read_file import ImzyReadFile
from depiction_io.imzy_backend.imzy_reader import ImzyReader
from depiction_io.ram.ram_read_file import RamReadFile
from depiction_io.ram.ram_reader import RamReader

# `_Writer` is private but it is what `RamWriteFile.writer()` yields, so it is the class that has
# to satisfy `GenericWriter`. Reaching for it by name beats reconstructing it from an instance.
from depiction_io.ram.ram_write_file import RamWriteFile, _Writer
from depiction_io.types import GenericReadFile, GenericReader, GenericWriteFile, GenericWriter

# `GenericWriter.open` is a classmethod taking a path. An in-memory sink has no path, and no
# holder of a `GenericWriter` ever calls it -- `GenericWriteFile.writer()` owns construction, so
# `open` is arguably on the wrong protocol. Exempted rather than faked.
EXEMPT: dict[tuple[str, str], set[str]] = {("GenericWriter", "_Writer"): {"open"}}

PAIRS = [
    (GenericReader, ImzyReader),
    (GenericReader, RamReader),
    (GenericReadFile, ImzyReadFile),
    (GenericReadFile, RamReadFile),
    (GenericWriter, ImzmlWriter),
    (GenericWriter, _Writer),
    (GenericWriteFile, ImzmlWriteFile),
    (GenericWriteFile, RamWriteFile),
]


def _public(obj: type) -> set[str]:
    return {name for name in dir(obj) if not name.startswith("_")}


@pytest.mark.parametrize(
    ("protocol", "implementation"),
    PAIRS,
    ids=[f"{protocol.__name__}-{implementation.__name__}" for protocol, implementation in PAIRS],
)
def test_implementation_exposes_every_protocol_member(protocol: type, implementation: type) -> None:
    exempt = EXEMPT.get((protocol.__name__, implementation.__name__), set())
    assert _public(protocol) - _public(implementation) - exempt == set()


@pytest.fixture
def ram_read_file() -> RamReadFile:
    """A read file built the way callers build one: through the writer.

    Constructing `RamReadFile` directly hid a defect for as long as it existed --
    `to_read_file()` handed it a *list* of coordinate arrays where it declares an `NDArray`, so
    `coordinates[:, :2]` raised `TypeError` on every read file the writer produced.
    """
    write_file = RamWriteFile(imzml_mode=ImzmlModeEnum.PROCESSED)
    with write_file.writer() as writer:
        writer.add_spectrum(np.array([100.0, 200.0, 300.0]), np.array([1.0, 2.0, 3.0]), (1, 4, 1))
        writer.add_spectrum(np.array([150.0, 250.0]), np.array([4.0, 5.0]), (2, 4, 1))
        writer.add_spectrum(np.array([120.0]), np.array([6.0]), (3, 5, 1))
    return write_file.to_read_file()


class TestRamReadFile:
    def test_coordinates_survive_the_writer(self, ram_read_file: RamReadFile) -> None:
        np.testing.assert_array_equal(np.array([[1, 4], [2, 4], [3, 5]]), ram_read_file.coordinates_2d)

    def test_coordinates_array_2d_is_labelled(self, ram_read_file: RamReadFile) -> None:
        result = ram_read_file.coordinates_array_2d
        assert result.dims == ("i", "d")
        assert list(result.coords["d"].values) == ["x", "y"]
        np.testing.assert_array_equal(np.array([[1, 4], [2, 4], [3, 5]]), result.values)

    def test_compact_metadata_describes_the_spectra_and_names_no_file(self, ram_read_file: RamReadFile) -> None:
        metadata = ram_read_file.compact_metadata
        assert metadata["n_spectra"] == 3
        assert metadata["imzml_mode"] == "PROCESSED"
        assert list(metadata["coordinate_extent"]) == [3, 2, 1]
        # `imzml_file` returns "/dev/null" here, which must not leak into a dict meant for
        # comparing acquisitions.
        assert "imzml_file" not in metadata
        assert "ibd_file" not in metadata

    def test_pixel_size_and_checksum_are_unknown_rather_than_invented(self, ram_read_file: RamReadFile) -> None:
        assert ram_read_file.pixel_size is None
        assert ram_read_file.is_checksum_valid is None

    def test_summary_reports_the_mode_and_count(self, ram_read_file: RamReadFile) -> None:
        summary = ram_read_file.summary()
        assert "PROCESSED" in summary
        assert "n_spectra: 3" in summary
        # accepted and ignored: there is no .ibd to verify either way
        assert ram_read_file.summary(checksums=False) == summary

    def test_print_summary_writes_the_summary(self, ram_read_file: RamReadFile, capsys: pytest.CaptureFixture) -> None:
        ram_read_file.print_summary()
        assert ram_read_file.summary() in capsys.readouterr().out


class TestRamReader:
    def test_get_spectrum_with_coords(self, ram_read_file: RamReadFile) -> None:
        with ram_read_file.reader() as reader:
            mz_arr, int_arr, coords = reader.get_spectrum_with_coords(1)
        np.testing.assert_array_equal(np.array([150.0, 250.0]), mz_arr)
        np.testing.assert_array_equal(np.array([4.0, 5.0]), int_arr)
        np.testing.assert_array_equal(np.array([2, 4, 1]), coords)

    def test_get_spectrum_coordinates(self, ram_read_file: RamReadFile) -> None:
        with ram_read_file.reader() as reader:
            np.testing.assert_array_equal(np.array([3, 5, 1]), reader.get_spectrum_coordinates(2))

    def test_coordinates_array_2d(self, ram_read_file: RamReadFile) -> None:
        with ram_read_file.reader() as reader:
            np.testing.assert_array_equal(np.array([[1, 4], [2, 4], [3, 5]]), reader.coordinates_array_2d.values)

    def test_get_spectra_mz_range_none_means_every_spectrum(self, ram_read_file: RamReadFile) -> None:
        with ram_read_file.reader() as reader:
            assert reader.get_spectra_mz_range(None) == reader.get_spectra_mz_range([0, 1, 2])


class TestRamWriter:
    def test_writer_reports_the_files_mode(self) -> None:
        write_file = RamWriteFile(imzml_mode=ImzmlModeEnum.CONTINUOUS)
        with write_file.writer() as writer:
            assert writer.imzml_mode == ImzmlModeEnum.CONTINUOUS

    def test_copy_spectra_round_trips_through_the_ram_backend(self, ram_read_file: RamReadFile) -> None:
        """The one member that reads and writes at once, so it exercises both halves together."""
        destination = RamWriteFile(imzml_mode=ImzmlModeEnum.PROCESSED)
        with ram_read_file.reader() as reader, destination.writer() as writer:
            writer.copy_spectra(reader, spectra_indices=[2, 0])

        copied = destination.to_read_file()
        assert copied.n_spectra == 2
        np.testing.assert_array_equal(np.array([[3, 5], [1, 4]]), copied.coordinates_2d)
        with copied.reader() as reader:
            np.testing.assert_array_equal(np.array([120.0]), reader.get_spectrum_mz(0))
            np.testing.assert_array_equal(np.array([100.0, 200.0, 300.0]), reader.get_spectrum_mz(1))
