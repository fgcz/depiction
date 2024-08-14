import numpy as np
import pytest

from depiction.tools.simulate.generate_synthetic_imzml import GenerateSyntheticImzml

height = 3
width = 5


@pytest.fixture()
def mock_rng(mocker):
    return mocker.MagicMock(name="mock_rng")


@pytest.fixture()
def generate_synthetic_imzml(mock_rng) -> GenerateSyntheticImzml:
    return GenerateSyntheticImzml(height, width, rng=mock_rng)


def test_generate_shift_map(mocker, mock_rng, generate_synthetic_imzml):
    # Setup
    mean = 0.3
    std = 0.1

    mock_rng.uniform.return_value = 0.123
    mock_perlin_noise = mocker.patch("depiction.tools.simulate.generate_synthetic_imzml.PerlinNoise")
    mock_noise_gen = mocker.MagicMock()

    class MockPerlin:
        def __init__(self):
            self._counter = 0

        def __call__(self, x):
            self._counter += 1
            return self._counter

    mock_noise_gen.side_effect = MockPerlin().__call__
    mock_perlin_noise.return_value = mock_noise_gen

    shift_map = generate_synthetic_imzml.generate_shift_map(mean, std)

    assert shift_map.shape == (height, width)
    np.testing.assert_almost_equal(shift_map.mean(), mean, decimal=2)
    np.testing.assert_almost_equal(shift_map.std(), std, decimal=2)

    expected_seed = round(1.23e11)
    mock_perlin_noise.assert_called_once_with(octaves=6, seed=expected_seed)


def test_get_shifted_target_masses(mock_rng, generate_synthetic_imzml):
    mock_rng.normal.return_value = np.array([0.001, -0.001, 0])
    mass_list = np.array([100.0, 200, 300])
    shift = 0.3
    shifted_masses = generate_synthetic_imzml.get_shifted_target_masses(mass_list, shift)
    np.testing.assert_array_almost_equal(shifted_masses, np.array([100.301, 200.299, 300.3]), decimal=8)
    mock_rng.normal.assert_called_once_with(scale=0.001, size=(3,))


if __name__ == "__main__":
    pytest.main([__file__])
