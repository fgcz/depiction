"""Tests for the default implementations in the ``GenericReader``/``GenericWriter`` seam.

``types.py`` is not just a set of signatures: it carries real behaviour that every
backend inherits (``get_spectra``, ``coordinates_2d``, ``get_spectra_mz_range``,
``copy_spectra``). None of it was covered, and all of it is about to be depended on by a
second backend, so it is tested here against minimal fakes rather than through a
concrete reader.
"""

from __future__ import annotations

from functools import cached_property

import numpy as np
import pytest
from numpy.typing import NDArray
from xarray import DataArray

from depiction.persistence import ImzmlModeEnum
from depiction.persistence.types import GenericReader, GenericWriter


class FakeReader(GenericReader):
    """Implements only the members a backend is actually required to provide."""

    def __init__(
        self,
        mz: list[NDArray[np.float64]],
        intensities: list[NDArray[np.float64]],
        coordinates: NDArray[np.int64],
        imzml_mode: ImzmlModeEnum,
    ) -> None:
        self._mz = mz
        self._int = intensities
        self._coordinates = coordinates
        self._imzml_mode = imzml_mode
        self.closed = False

    def close(self) -> None:
        self.closed = True

    @property
    def imzml_mode(self) -> ImzmlModeEnum:
        return self._imzml_mode

    @property
    def n_spectra(self) -> int:
        return len(self._mz)

    @cached_property
    def coordinates(self) -> NDArray[np.int64]:
        return self._coordinates

    def get_spectrum_mz(self, i_spectrum: int) -> NDArray[np.float64]:
        return self._mz[i_spectrum]

    def get_spectrum_int(self, i_spectrum: int) -> NDArray[np.float64]:
        return self._int[i_spectrum]


class RecordingWriter(GenericWriter):
    """Captures what ``copy_spectra`` forwards to ``add_spectrum``."""

    def __init__(self, imzml_mode: ImzmlModeEnum) -> None:
        self._imzml_mode = imzml_mode
        self.added: list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]] = []

    @property
    def imzml_mode(self) -> ImzmlModeEnum:
        return self._imzml_mode

    def close(self) -> None:
        pass

    def add_spectrum(self, mz_arr, int_arr, coordinates) -> None:  # noqa: ANN001
        self.added.append((mz_arr, int_arr, coordinates))


@pytest.fixture
def processed_reader() -> FakeReader:
    return FakeReader(
        mz=[np.array([100.0, 200.0, 300.0]), np.array([150.0, 250.0]), np.array([120.0])],
        intensities=[np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0]), np.array([6.0])],
        coordinates=np.array([[1, 1, 5], [2, 1, 5], [3, 1, 5]], dtype=np.int64),
        imzml_mode=ImzmlModeEnum.PROCESSED,
    )


@pytest.fixture
def continuous_reader() -> FakeReader:
    shared = np.array([100.0, 200.0, 300.0])
    return FakeReader(
        mz=[shared, shared, shared],
        intensities=[np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0, 6.0]), np.array([7.0, 8.0, 9.0])],
        coordinates=np.array([[1, 1], [2, 1], [3, 1]], dtype=np.int64),
        imzml_mode=ImzmlModeEnum.CONTINUOUS,
    )


class TestCoordinates:
    def test_coordinates_2d_truncates_the_third_dimension(self, processed_reader: FakeReader) -> None:
        np.testing.assert_array_equal(np.array([[1, 1], [2, 1], [3, 1]]), processed_reader.coordinates_2d)

    def test_coordinates_2d_is_a_noop_for_2d_input(self, continuous_reader: FakeReader) -> None:
        np.testing.assert_array_equal(continuous_reader.coordinates, continuous_reader.coordinates_2d)

    def test_coordinates_array_2d_is_labelled(self, processed_reader: FakeReader) -> None:
        result = processed_reader.coordinates_array_2d
        assert isinstance(result, DataArray)
        assert result.dims == ("i", "d")
        assert list(result.coords["d"].values) == ["x", "y"]
        np.testing.assert_array_equal(np.array([[1, 1], [2, 1], [3, 1]]), result.values)

    def test_get_spectrum_coordinates(self, processed_reader: FakeReader) -> None:
        np.testing.assert_array_equal(np.array([2, 1, 5]), processed_reader.get_spectrum_coordinates(1))


class TestGetSpectra:
    def test_processed_mode_returns_ragged_tuples(self, processed_reader: FakeReader) -> None:
        mz_arrays, int_arrays = processed_reader.get_spectra([0, 1, 2])
        assert [len(arr) for arr in mz_arrays] == [3, 2, 1]
        np.testing.assert_array_equal(np.array([150.0, 250.0]), mz_arrays[1])
        np.testing.assert_array_equal(np.array([4.0, 5.0]), int_arrays[1])

    def test_continuous_mode_stacks_into_rectangular_arrays(self, continuous_reader: FakeReader) -> None:
        mz_arrays, int_arrays = continuous_reader.get_spectra([0, 1, 2])
        assert mz_arrays.shape == (3, 3)
        assert int_arrays.shape == (3, 3)
        # every row of the m/z block is the same shared axis
        np.testing.assert_array_equal(np.tile(np.array([100.0, 200.0, 300.0]), (3, 1)), mz_arrays)
        np.testing.assert_array_equal(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]), int_arrays)

    def test_continuous_mode_honours_the_requested_subset(self, continuous_reader: FakeReader) -> None:
        mz_arrays, int_arrays = continuous_reader.get_spectra([2, 0])
        assert mz_arrays.shape == (2, 3)
        np.testing.assert_array_equal(np.array([[7.0, 8.0, 9.0], [1.0, 2.0, 3.0]]), int_arrays)


class TestSpectrumAccessors:
    def test_get_spectrum_returns_both_arrays(self, processed_reader: FakeReader) -> None:
        mz_arr, int_arr = processed_reader.get_spectrum(0)
        np.testing.assert_array_equal(np.array([100.0, 200.0, 300.0]), mz_arr)
        np.testing.assert_array_equal(np.array([1.0, 2.0, 3.0]), int_arr)

    def test_get_spectrum_with_coords(self, processed_reader: FakeReader) -> None:
        mz_arr, int_arr, coords = processed_reader.get_spectrum_with_coords(1)
        np.testing.assert_array_equal(np.array([150.0, 250.0]), mz_arr)
        np.testing.assert_array_equal(np.array([4.0, 5.0]), int_arr)
        np.testing.assert_array_equal(np.array([2, 1, 5]), coords)

    def test_get_spectrum_n_points(self, processed_reader: FakeReader) -> None:
        assert [processed_reader.get_spectrum_n_points(i) for i in range(3)] == [3, 2, 1]


class TestGetSpectraMzRange:
    def test_spans_all_requested_spectra(self, processed_reader: FakeReader) -> None:
        assert processed_reader.get_spectra_mz_range([0, 1, 2]) == (100.0, 300.0)

    def test_subset_narrows_the_range(self, processed_reader: FakeReader) -> None:
        assert processed_reader.get_spectra_mz_range([1, 2]) == (120.0, 250.0)

    def test_none_means_every_spectrum(self, processed_reader: FakeReader) -> None:
        assert processed_reader.get_spectra_mz_range(None) == processed_reader.get_spectra_mz_range([0, 1, 2])


class TestContextManager:
    def test_exit_closes_the_reader(self, processed_reader: FakeReader) -> None:
        with processed_reader as reader:
            assert reader is processed_reader
            assert not reader.closed
        assert processed_reader.closed

    def test_exit_closes_on_exception(self, processed_reader: FakeReader) -> None:
        with pytest.raises(RuntimeError), processed_reader:
            raise RuntimeError("boom")
        assert processed_reader.closed


class TestCopySpectra:
    def test_forwards_every_requested_spectrum_with_its_coordinates(self, processed_reader: FakeReader) -> None:
        writer = RecordingWriter(ImzmlModeEnum.PROCESSED)
        writer.copy_spectra(processed_reader, spectra_indices=[0, 2])

        assert len(writer.added) == 2
        np.testing.assert_array_equal(np.array([100.0, 200.0, 300.0]), writer.added[0][0])
        np.testing.assert_array_equal(np.array([1, 1, 5]), writer.added[0][2])
        np.testing.assert_array_equal(np.array([120.0]), writer.added[1][0])
        np.testing.assert_array_equal(np.array([3, 1, 5]), writer.added[1][2])

    def test_preserves_the_requested_order(self, processed_reader: FakeReader) -> None:
        writer = RecordingWriter(ImzmlModeEnum.PROCESSED)
        writer.copy_spectra(processed_reader, spectra_indices=[2, 0, 1])
        assert [len(mz) for mz, _, _ in writer.added] == [1, 3, 2]

    def test_tqdm_position_does_not_change_what_is_written(self, processed_reader: FakeReader) -> None:
        plain = RecordingWriter(ImzmlModeEnum.PROCESSED)
        with_progress = RecordingWriter(ImzmlModeEnum.PROCESSED)
        plain.copy_spectra(processed_reader, spectra_indices=[0, 1, 2])
        with_progress.copy_spectra(processed_reader, spectra_indices=[0, 1, 2], tqdm_position=1)

        assert len(plain.added) == len(with_progress.added)
        for (mz_a, int_a, coords_a), (mz_b, int_b, coords_b) in zip(plain.added, with_progress.added):
            np.testing.assert_array_equal(mz_a, mz_b)
            np.testing.assert_array_equal(int_a, int_b)
            np.testing.assert_array_equal(coords_a, coords_b)
