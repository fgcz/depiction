import numpy as np
from perlin_noise import PerlinNoise


class GenerateSyntheticImzml:
    def __init__(self, height: int, width: int, rng: np.random.Generator) -> None:
        self._height = height
        self._width = width
        self._rng = rng

    def generate_shift_map(self, mean: float = 0.3, std: float = 0.1):
        """Generates a mass shift map"""
        seed = round(self._rng.uniform() * 1e12)
        noise_gen = PerlinNoise(octaves=6, seed=seed)
        noise_2d = np.asarray(
            [[noise_gen([i / self._width, j / self._height]) for j in range(self._width)] for i in range(self._height)]
        )
        # adjust the values to the desired mean and std
        noise_2d = (noise_2d - noise_2d.mean()) / noise_2d.std() * std + mean
        return noise_2d
