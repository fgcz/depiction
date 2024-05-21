import argparse

import logging
import os
from typing import Optional

import numpy as np
import pyimzml.ImzMLParser
import pyimzml.ImzMLWriter
from tqdm import tqdm

from ionplotter.persistence.imzml_read_file import ImzmlReadFile
from ionplotter.persistence.imzml_write_file import ImzmlWriteFile


# TODO under development
class SplitImzml:
    def __init__(
        self,
        source_imzml_path: str,
        n_parts: Optional[int],
        n_spectra_per_part: Optional[int],
    ) -> None:
        self._source_imzml_path = source_imzml_path
        self._n_parts = n_parts
        self._n_spectra_per_part = n_spectra_per_part

    def _get_split_indices(self) -> list[np.ndarray[int]]:
        """Returns a list with the indices per split part."""
        n_spectra_file = len(pyimzml.ImzMLParser.ImzMLParser(self._source_imzml_path).coordinates)
        if self._n_parts and self._n_spectra_per_part:
            raise ValueError("Only one of n_parts and n_spectra can be provided.")
        n_parts = self._n_parts if self._n_parts else max(n_spectra_file // self._n_spectra_per_part, 1)
        return np.array_split(np.arange(n_spectra_file), n_parts)

    def write_splits(self, output_dir: str) -> dict:
        split_indices = self._get_split_indices()
        os.makedirs(output_dir, exist_ok=True)
        self._logger.info(f"Splitting file into {len(split_indices)} parts.")

        output_files = []  # type: list[str]
        output_spectra_indices = []  # type: list[np.ndarray[int]]
        parser = pyimzml.ImzMLParser.ImzMLParser(self._source_imzml_path)

        file_params = parser.metadata.file_description.param_by_name
        imzml_mode = "processed" if file_params.get("processed", False) else "continuous"

        for i_part, indices in tqdm(enumerate(split_indices), desc=" part", position=0):
            output_spectra_indices.append(indices)
            filename = os.path.join(output_dir, f"part_{i_part}.imzML")
            output_files.append(filename)

            with pyimzml.ImzMLWriter.ImzMLWriter(filename, mode=imzml_mode) as writer:
                for index in tqdm(indices, desc="  spectrum", position=1):
                    writer.addSpectrum(*parser.getspectrum(index), parser.coordinates[index])

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)


class ImzmlSplitter:
    def __init__(
        self,
        read_file: ImzmlReadFile,
        n_parts: Optional[int],
        n_spectra_per_part: Optional[int],
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
        os.makedirs(output_dir, exist_ok=True)
        self._logger.info(f"Splitting file into {len(split_indices)} parts.")

        output_files = []  # type: list[str]
        output_spectra_indices = []  # type: list[np.ndarray[int]]

        with self._read_file.reader() as reader:
            mz_is_unique = True

            for i_part, indices in tqdm(enumerate(split_indices), desc=" part", position=0):
                output_spectra_indices.append(indices)
                filename = os.path.join(output_dir, f"part_{i_part}.imzML")
                output_files.append(filename)
                with ImzmlWriteFile(path=filename, imzml_mode=self._read_file.imzml_mode).writer() as writer:
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
    read_file = ImzmlReadFile(input_imzml)
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
