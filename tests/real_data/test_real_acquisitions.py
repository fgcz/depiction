"""Reads two real third-party acquisitions and checks what comes back.

These are the only files in the repository's reach that were produced by somebody else's
instrument and somebody else's export software. That is the whole point of them. The
differential corpus in `tests/differential/` is thorough about dtypes, modes and edge
cases, but every file in it was written by imzy's own writer, so a header convention that
imzy emits and misreads symmetrically is invisible there. A SCiLS Lab export and a Cardinal
export are not.

They skip when `.test-data/` is empty, which is the normal case; see
`tests/real_data/fetch.py`. Assertions come from `datasets.ExpectedReading`, whose
provenance is documented in `datasets.py` -- briefly: measured through both the hand-rolled
parser and imzy while both existed, and they agreed.

Reading goes through `get_read_file`, never through a named backend class, for the same
reason `test_reader_parity.py` does it: the seam is the public entry point and the thing
every tool actually calls.
"""

from __future__ import annotations

import hashlib
import pickle
from typing import TYPE_CHECKING

import numpy as np
import pytest

from depiction.parallel_ops import ParallelConfig, ReadSpectraParallel
from depiction_io import ImzmlModeEnum
from depiction_io.imzy_backend.imzml_scan import scan_imzml

if TYPE_CHECKING:
    from pathlib import Path

    from depiction_io.types import GenericReadFile, GenericReader
    from tests.real_data.datasets import PublicDataset

pytestmark = pytest.mark.real_data

#: `IMS:1000052`, position z. Searched for in the raw XML so that the expectation about
#: `coordinates` is derived from the file rather than from the code under test.
_POSITION_Z = b'accession="IMS:1000052"'


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class TestIntegrity:
    """Two different questions: is this the published file, and does it match its own header."""

    def test_sha256_matches_the_deposit(self, dataset: PublicDataset) -> None:
        for remote in dataset.files:
            assert _sha256(remote.local_path) == remote.sha256, remote.filename

    def test_declared_ibd_checksum_validates(self, read_file: GenericReadFile) -> None:
        # `ParseMetadata` is the one piece of the deleted parser that Phase E kept, and it
        # was kept for exactly this -- imzy parses no checksums at all. Nothing else
        # exercises it against a file it did not write.
        assert read_file.is_checksum_valid is True


class TestScan:
    """`scan_imzml` guards the reader; its only real-file coverage is a three-chunk fragment."""

    def test_no_unsupported_compression(self, dataset: PublicDataset) -> None:
        scan = scan_imzml(dataset.imzml.local_path)
        assert scan.unsupported_compression is None

    def test_no_encoded_lengths_are_collected(self, dataset: PublicDataset) -> None:
        # Both files are uncompressed, so the zlib path must not engage. A scan that
        # collected offsets here would send every read through `ZlibIMZMLReader`.
        assert scan_imzml(dataset.imzml.local_path).encoded_lengths is None


class TestGeometry:
    def test_n_spectra(self, read_file: GenericReadFile, dataset: PublicDataset) -> None:
        assert read_file.n_spectra == dataset.expected.n_spectra

    def test_imzml_mode(self, read_file: GenericReadFile, dataset: PublicDataset) -> None:
        assert read_file.imzml_mode == ImzmlModeEnum[dataset.expected.imzml_mode]

    def test_coordinates_have_z_exactly_when_the_imzml_declares_it(
        self, read_file: GenericReadFile, dataset: PublicDataset
    ) -> None:
        # imzy always reports three columns; `declares_z` is what stops a 2D file from
        # gaining one. Checked against the XML, not against the scan that produced it.
        declares_z = _POSITION_Z in dataset.imzml.local_path.read_bytes()
        assert read_file.coordinates.shape == (dataset.expected.n_spectra, 3 if declares_z else 2)

    def test_coordinate_extent(self, read_file: GenericReadFile, dataset: PublicDataset) -> None:
        coordinates = read_file.coordinates_2d
        extent = tuple(coordinates.max(axis=0) - coordinates.min(axis=0) + 1)
        assert extent == dataset.expected.coordinate_extent

    def test_pixel_size(self, read_file: GenericReadFile, dataset: PublicDataset) -> None:
        expected = dataset.expected.pixel_size_um
        if expected is None:
            assert read_file.pixel_size is None
            return
        assert read_file.pixel_size is not None
        assert (read_file.pixel_size.size_x, read_file.pixel_size.size_y) == (expected, expected)
        assert read_file.pixel_size.unit == "micrometer"


class TestSpectra:
    def test_dtypes(self, read_file: GenericReadFile, dataset: PublicDataset) -> None:
        with read_file.reader() as reader:
            mz_arr, int_arr = reader.get_spectrum(0)
        assert mz_arr.dtype == np.dtype(dataset.expected.mz_dtype)
        assert int_arr.dtype == np.dtype(dataset.expected.int_dtype)

    def test_mz_axis_bounds_and_bins(self, read_file: GenericReadFile, dataset: PublicDataset) -> None:
        expected = dataset.expected
        with read_file.reader() as reader:
            mz_arr = reader.get_spectrum_mz(0)
        assert len(mz_arr) == expected.n_bins
        assert mz_arr[0] == pytest.approx(expected.mz_min, abs=expected.mz_tolerance)
        assert mz_arr[-1] == pytest.approx(expected.mz_max, abs=expected.mz_tolerance)

    def test_mz_axis_is_sorted(self, read_file: GenericReadFile) -> None:
        # Not pinned to a recorded number, and true of any acquisition: every tool
        # downstream -- binning, peak picking, calibration -- assumes it.
        with read_file.reader() as reader:
            mz_arr = reader.get_spectrum_mz(0)
        assert np.all(np.diff(mz_arr) > 0)

    def test_arrays_are_paired(
        self, read_file: GenericReadFile, dataset: PublicDataset, sample_indices: list[int]
    ) -> None:
        with read_file.reader() as reader:
            for i in sample_indices:
                mz_arr, int_arr = reader.get_spectrum(i)
                assert len(mz_arr) == len(int_arr) == dataset.expected.n_bins, f"spectrum {i}"

    def test_continuous_files_share_one_mz_axis(self, read_file: GenericReadFile, sample_indices: list[int]) -> None:
        with read_file.reader() as reader:
            first = reader.get_spectrum_mz(sample_indices[0])
            for i in sample_indices[1:]:
                np.testing.assert_array_equal(first, reader.get_spectrum_mz(i), err_msg=f"spectrum {i}")

    def test_get_spectrum_n_points_matches_the_array(self, read_file: GenericReadFile, dataset: PublicDataset) -> None:
        # This one changed meaning in Phase E: the deleted parser reported the encoded
        # length in bytes, four times too large for a float32 array. Its only caller is
        # untested, so this is where the new answer gets checked against reality.
        with read_file.reader() as reader:
            assert reader.get_spectrum_n_points(0) == dataset.expected.n_bins


class TestReadPaths:
    """The batched, pickled and parallel paths, on files big enough for them to matter."""

    def test_batched_read_agrees_with_single_reads(self, read_file: GenericReadFile, sample_indices: list[int]) -> None:
        # `ImzyReader.get_spectra` reassembles continuous-mode output by hand, repeating a
        # single m/z axis across the batch. Only synthetic files have exercised that.
        with read_file.reader() as reader:
            mz_arrays, int_arrays = reader.get_spectra(sample_indices)
            for position, i in enumerate(sample_indices):
                np.testing.assert_array_equal(reader.get_spectrum_mz(i), mz_arrays[position], err_msg=f"mz {i}")
                np.testing.assert_array_equal(reader.get_spectrum_int(i), int_arrays[position], err_msg=f"int {i}")

    def test_survives_a_pickle_round_trip(self, read_file: GenericReadFile, sample_indices: list[int]) -> None:
        restored = pickle.loads(pickle.dumps(read_file))
        assert restored.n_spectra == read_file.n_spectra
        np.testing.assert_array_equal(read_file.coordinates, restored.coordinates)
        with read_file.reader() as reader, restored.reader() as restored_reader:
            for i in sample_indices:
                np.testing.assert_array_equal(
                    reader.get_spectrum_int(i), restored_reader.get_spectrum_int(i), err_msg=f"spectrum {i}"
                )

    def test_read_spectra_parallel_agrees_with_serial(
        self, read_file: GenericReadFile, sample_indices: list[int]
    ) -> None:
        parallel = ReadSpectraParallel.from_config(ParallelConfig(n_jobs=2, task_size=2))
        actual = parallel.map_chunked(
            read_file=read_file,
            operation=_sum_intensities,
            spectra_indices=np.array(sample_indices),
            reduce_fn=ReadSpectraParallel.reduce_concat,
        )
        with read_file.reader() as reader:
            expected = _sum_intensities(reader, sample_indices)
        assert actual == expected


def _sum_intensities(reader: GenericReader, spectra_ids: list[int]) -> list[float]:
    """Module level so it can be pickled into the worker processes."""
    return [float(reader.get_spectrum_int(i).sum()) for i in spectra_ids]


def test_summary_describes_the_file(read_file: GenericReadFile, dataset: PublicDataset) -> None:
    # Checksums off: `is_checksum_valid` hashes the whole .ibd and has its own test.
    summary = read_file.summary(checksums=False)
    assert dataset.imzml.filename in summary
    assert f"n_spectra: {dataset.expected.n_spectra}" in summary
    assert f"({dataset.expected.n_bins} bins)" in summary
