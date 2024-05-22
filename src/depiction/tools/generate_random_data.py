import numpy as np
from tqdm import tqdm


class GenerateRandomData:
    def __init__(self, write_file, seed: int = 0) -> None:
        self._write_file = write_file
        self._rng = np.random.default_rng(seed=seed)
        self._coordinate_index = 0

    def generate_random_data(self, n_spectra: int, n_mz: int, continuous: bool) -> None:
        """Generates random uniform data and writes it into spectra."""
        if continuous:
            mz_arr = np.sort(self._rng.uniform(low=0, high=1000, size=n_mz))
        for _ in tqdm(range(n_spectra)):
            if not continuous:
                mz_arr = np.sort(self._rng.uniform(low=0, high=1000, size=n_mz))
            int_arr = self._rng.uniform(low=0, high=1000, size=n_mz)
            self._write_file.add_spectrum(mz_arr, int_arr, self._get_next_coordinate())

    def _get_next_coordinate(self) -> tuple[int, int]:
        self._coordinate_index = self._coordinate_index + 1
        x = self._coordinate_index % 25
        y = self._coordinate_index // 25
        return x, y
