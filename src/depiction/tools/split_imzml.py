import argparse

import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

from depiction_io import GenericReadFile, ImzmlWriteFile, get_read_file


class ImzmlSplitter:
    def __init__(
        self,
        read_file: GenericReadFile,
        n_parts: int | None,
        n_spectra_per_part: int | None,
    ) -> None:
        self._read_file = read_file
        self._n_parts = n_parts
        self._n_spectra_per_part = n_spectra_per_part

    def get_split_indices(self) -> list[np.ndarray[int]]:
        """Returns a list with the indices per split part."""
        if self._n_parts and self._n_spectra_per_part:
            raise ValueError("Only one of n_parts and n_spectra can be provided.")
        n_parts = self._n_parts if self._n_parts else max(self._read_file.n_spectra // self._n_spectra_per_part, 1)
        return np.array_split(np.arange(self._read_file.n_spectra), n_parts)

    def write_splits(self, output_dir: str) -> dict:
        split_indices = self.get_split_indices()
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self._logger.info(f"Splitting file into {len(split_indices)} parts.")

        output_files = []  # type: list[str]
        output_spectra_indices = []  # type: list[np.ndarray[int]]

        with self._read_file.reader() as reader:
            mz_is_unique = True

            for i_part, indices in tqdm(enumerate(split_indices), desc=" part", position=0):
                output_spectra_indices.append(indices)
                filename = str(Path(output_dir) / f"part_{i_part}.imzML")
                output_files.append(filename)
                with ImzmlWriteFile(
                    path=filename,
                    imzml_mode=self._read_file.imzml_mode,
                    pixel_size=self._read_file.pixel_size,
                ).writer() as writer:
                    # writer.deactivate_alignment_tracker()
                    writer.copy_spectra(reader, spectra_indices=indices, tqdm_position=1)
                    mz_is_unique = mz_is_unique and writer.is_aligned

        return {
            "output_files": output_files,
            "output_spectra_indices": output_spectra_indices,
            "mz_is_unique": mz_is_unique,
        }

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)


def main_split_imzml(input_imzml: str, output_dir: str, n_parts: int, n_spectra: int) -> list[str]:
    """
    Splits the imzml file in ``input_imzml`` into parts stored in ``output_dir``.
    Either n_parts or n_spectra must be provided.
    If n_parts is provided, the file is split into n_parts parts of equal size.
    If n_spectra is provided, the file is split into parts of size n_spectra.
    :return: The paths to the created files.
    """
    read_file = get_read_file(input_imzml)
    splitter = ImzmlSplitter(read_file, n_parts=n_parts, n_spectra_per_part=n_spectra)
    result = splitter.write_splits(output_dir)
    return result["output_files"]


def main() -> None:
    """Invokes CLI for main_split_imzml."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("input_imzml", type=str)
    parser.add_argument("output_dir", type=str)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--n_parts", type=int, help="Number of parts to split into.", default=None)
    group.add_argument("--n_spectra", type=int, help="Number of spectra per part.", default=None)

    args = vars(parser.parse_args())
    main_split_imzml(**args)


if __name__ == "__main__":
    main()
