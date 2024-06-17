import numpy as np
import pytest

from depiction.tools.simulate import SyntheticMSIDataGenerator


def test_get_mz_arr() -> None:
    min_mass = 100
    max_mass = 5000
    mz_arr = SyntheticMSIDataGenerator.get_mz_arr(min_mass, max_mass, bin_width_ppm=1000000)
    expected_arr = np.array([100., 258.06451613, 574.19354839, 1206.4516129, 2470.96774194, 5000.])
    np.testing.assert_array_almost_equal(mz_arr, expected_arr)


def test_get_baseline() -> None:
    points = SyntheticMSIDataGenerator.get_baseline(5, 3.0)
    assert points.shape == (5,)
    expected_points = np.array([3., 1.4171, 0.66939, 0.316198, 0.149361])
    np.testing.assert_array_almost_equal(points, expected_points, decimal=4)


if __name__ == "__main__":
    pytest.main()
