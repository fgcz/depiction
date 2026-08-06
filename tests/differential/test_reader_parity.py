"""Differential tests over the ``GenericReader`` seam.

Every assertion here is phrased against the protocol in ``depiction_io.types``,
never against a concrete backend. That is deliberate: when a second imzML backend is
added, it gets registered in ``READ_FILE_BACKENDS`` and this file becomes a genuine
A/B comparison without any of the assertions changing.

The file earns its keep four ways:

1. round-trip -- what the writer wrote is what the reader reads, per dtype and mode;
2. cross-implementation -- ``ImzmlReadFile`` must agree with ``RamReadFile``;
3. A/B -- every assertion runs against both the legacy parser and the imzy backend, and
   they must be indistinguishable;
4. compression -- a zlib file must read identically to the uncompressed file it was
   derived from, which is the check that makes swapping in a backend without zlib
   support fail loudly instead of silently returning noise.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable

import numpy as np
import pytest

from depiction_io import ImzmlModeEnum, ImzmlReadFile, ImzyReadFile, RamReadFile
from depiction_io.imzy_backend import UnsupportedCompressionError
from depiction_io.types import GenericReadFile
from tests.differential.corpus import Case

#: Backends that can open a corpus case from disk. Every test below runs once per entry,
#: which is what makes this an A/B comparison rather than a round-trip check.
READ_FILE_BACKENDS: dict[str, Callable[[Case], GenericReadFile]] = {
    "imzml": lambda case: ImzmlReadFile(case.path),
    "imzy": lambda case: ImzyReadFile(case.path),
}

#: Backends that refuse compressed input instead of reading it. See `TestCompression`.
BACKENDS_WITHOUT_COMPRESSION = {"imzy"}


@pytest.fixture(params=sorted(READ_FILE_BACKENDS))
def backend(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def read_file(backend: str, case: Case) -> GenericReadFile:
    """A read file the backend can actually read.

    A backend that does not support compression is skipped for the compressed cases rather
    than xfailed, because "it raises" is not a weaker form of "it reads" -- it is the
    intended behaviour, and `TestCompression` asserts it directly.
    """
    if case.compressed and backend in BACKENDS_WITHOUT_COMPRESSION:
        pytest.skip(f"{backend} refuses compressed input; see TestCompression")
    return READ_FILE_BACKENDS[backend](case)


def _ram_read_file(case: Case) -> RamReadFile:
    """An in-memory reader over the same source data, used as the comparison backend."""
    return RamReadFile(
        mz_arr_list=case.expected_mz,
        int_arr_list=case.expected_int,
        coordinates=case.spectra.coordinates,
    )


class TestRoundTrip:
    """What the writer wrote is what the reader reads."""

    def test_n_spectra(self, read_file: GenericReadFile, case: Case) -> None:
        assert read_file.n_spectra == case.spectra.n_spectra

    def test_imzml_mode(self, read_file: GenericReadFile, case: Case) -> None:
        assert read_file.imzml_mode == case.expected_read_mode

    def test_coordinates(self, read_file: GenericReadFile, case: Case) -> None:
        np.testing.assert_array_equal(case.spectra.coordinates, read_file.coordinates)

    def test_coordinates_keep_all_dimensions(self, read_file: GenericReadFile, case: Case) -> None:
        # Regression guard: a truthiness test on the position-z Element used to drop the
        # third dimension of every 3D file silently.
        assert read_file.coordinates.shape == case.spectra.coordinates.shape

    def test_spectrum_mz(self, read_file: GenericReadFile, case: Case) -> None:
        with read_file.reader() as reader:
            for i, expected in enumerate(case.expected_mz):
                np.testing.assert_array_equal(expected, reader.get_spectrum_mz(i), err_msg=f"spectrum {i}")

    def test_spectrum_int(self, read_file: GenericReadFile, case: Case) -> None:
        with read_file.reader() as reader:
            for i, expected in enumerate(case.expected_int):
                np.testing.assert_array_equal(expected, reader.get_spectrum_int(i), err_msg=f"spectrum {i}")

    def test_dtypes_are_preserved(self, read_file: GenericReadFile, case: Case) -> None:
        with read_file.reader() as reader:
            assert reader.get_spectrum_mz(0).dtype == np.dtype(case.mz_dtype)
            assert reader.get_spectrum_int(0).dtype == np.dtype(case.int_dtype)

    def test_get_spectra_matches_individual_reads(self, read_file: GenericReadFile, case: Case) -> None:
        indices = list(range(case.spectra.n_spectra))
        with read_file.reader() as reader:
            mz_arrays, int_arrays = reader.get_spectra(indices)
            for i in indices:
                np.testing.assert_array_equal(reader.get_spectrum_mz(i), mz_arrays[i], err_msg=f"mz {i}")
                np.testing.assert_array_equal(reader.get_spectrum_int(i), int_arrays[i], err_msg=f"int {i}")

    def test_survives_a_pickle_round_trip(self, read_file: GenericReadFile, case: Case) -> None:
        # `ReadSpectraParallel` pickles the read file into every worker process and opens a
        # reader there, so a backend whose reader cannot be reconstructed from its state is
        # unusable no matter how well it reads in-process.
        restored = pickle.loads(pickle.dumps(read_file))
        assert restored.n_spectra == read_file.n_spectra
        np.testing.assert_array_equal(read_file.coordinates, restored.coordinates)
        with restored.reader() as reader:
            restored_reader = pickle.loads(pickle.dumps(reader))
            for i, expected in enumerate(case.expected_mz):
                np.testing.assert_array_equal(expected, restored_reader.get_spectrum_mz(i), err_msg=f"spectrum {i}")

    def test_continuous_files_share_one_mz_axis(self, read_file: GenericReadFile, case: Case) -> None:
        if case.expected_read_mode != ImzmlModeEnum.CONTINUOUS:
            pytest.skip("only meaningful for continuous mode")
        with read_file.reader() as reader:
            first = reader.get_spectrum_mz(0)
            for i in range(1, case.spectra.n_spectra):
                np.testing.assert_array_equal(first, reader.get_spectrum_mz(i), err_msg=f"spectrum {i}")


class TestCrossImplementation:
    """The on-disk reader and the in-memory reader must be indistinguishable."""

    def test_metadata_agrees(self, read_file: GenericReadFile, case: Case) -> None:
        ram = _ram_read_file(case)
        assert read_file.n_spectra == ram.n_spectra
        assert read_file.imzml_mode == ram.imzml_mode
        np.testing.assert_array_equal(ram.coordinates, read_file.coordinates)

    def test_spectra_agree(self, read_file: GenericReadFile, case: Case) -> None:
        ram = _ram_read_file(case)
        with read_file.reader() as reader, ram.reader() as ram_reader:
            for i in range(case.spectra.n_spectra):
                np.testing.assert_array_equal(
                    ram_reader.get_spectrum_mz(i), reader.get_spectrum_mz(i), err_msg=f"mz {i}"
                )
                np.testing.assert_array_equal(
                    ram_reader.get_spectrum_int(i), reader.get_spectrum_int(i), err_msg=f"int {i}"
                )

    def test_coordinates_array_2d_agrees(self, read_file: GenericReadFile, case: Case) -> None:
        ram = _ram_read_file(case)
        np.testing.assert_array_equal(ram.coordinates_2d, read_file.coordinates_2d)


@pytest.mark.compressed_only
class TestCompression:
    """A zlib file must be indistinguishable from the uncompressed file it came from.

    Without this, a backend that ignores the compression cvParam reads a compressed .ibd
    as raw bytes and returns plausible-looking noise rather than raising.
    """

    def test_declares_zlib_in_the_imzml(self, case: Case) -> None:
        assert case.compressed
        assert "MS:1000574" in case.path.read_text()

    def test_ibd_is_actually_smaller_or_different(self, case: Case, corpus: dict[str, Case]) -> None:
        source = corpus[case.derived_from]
        assert case.path.with_suffix(".ibd").read_bytes() != source.path.with_suffix(".ibd").read_bytes()

    def test_backends_without_compression_refuse_to_read(self, case: Case, backend: str) -> None:
        # The point of the guard, and the reason the `read_file` fixture is allowed to skip
        # these cases elsewhere: a backend that cannot decompress must say so rather than
        # read the .ibd as raw floats and return noise.
        if backend not in BACKENDS_WITHOUT_COMPRESSION:
            pytest.skip(f"{backend} supports compressed input")
        read_file = READ_FILE_BACKENDS[backend](case)
        with pytest.raises(UnsupportedCompressionError):
            read_file.n_spectra

    def test_reads_identically_to_uncompressed_twin(
        self, read_file: GenericReadFile, case: Case, corpus: dict[str, Case], backend: str
    ) -> None:
        source = READ_FILE_BACKENDS[backend](corpus[case.derived_from])
        assert read_file.n_spectra == source.n_spectra
        assert read_file.imzml_mode == source.imzml_mode
        np.testing.assert_array_equal(source.coordinates, read_file.coordinates)
        with read_file.reader() as reader, source.reader() as source_reader:
            for i in range(source.n_spectra):
                np.testing.assert_array_equal(
                    source_reader.get_spectrum_mz(i), reader.get_spectrum_mz(i), err_msg=f"mz {i}"
                )
                np.testing.assert_array_equal(
                    source_reader.get_spectrum_int(i), reader.get_spectrum_int(i), err_msg=f"int {i}"
                )

    def test_checksum_is_still_valid(self, case: Case, backend: str) -> None:
        # Guards the corpus builder itself: recompressing rewrites the .ibd, so the
        # SHA-1 in the imzML has to be recomputed or every compressed case would carry a
        # spurious checksum failure. Built directly rather than through `read_file`, since
        # verifying a checksum needs no decompression and both backends must manage it.
        assert READ_FILE_BACKENDS[backend](case).is_checksum_valid is not False
