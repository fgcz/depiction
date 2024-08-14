import numpy as np
import pytest

from depiction.tools.simulate.generate_synthetic_imzml import GenerateSyntheticImzml


def test_generate_shift_map(mocker):
    # Setup
    height = 3
    width = 5
    mean = 0.3
    std = 0.1

    mock_rng = mocker.MagicMock(name="mock_rng", uniform=mocker.MagicMock(return_value=0.123))
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

    synth_imzml = GenerateSyntheticImzml(height, width, rng=mock_rng)
    shift_map = synth_imzml.generate_shift_map(mean, std)

    assert shift_map.shape == (height, width)

    np.testing.assert_almost_equal(shift_map.mean(), mean, decimal=2)
    np.testing.assert_almost_equal(shift_map.std(), std, decimal=2)

    expected_seed = round(1.23e11)
    mock_perlin_noise.assert_called_once_with(octaves=6, seed=expected_seed)


if __name__ == "__main__":
    pytest.main([__file__])
