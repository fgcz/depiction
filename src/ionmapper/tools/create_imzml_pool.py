from collections import defaultdict
from functools import cached_property

import numpy as np
import pandas as pd
import os
from numpy.typing import NDArray
from tqdm import tqdm

from ionmapper.persistence import ImzmlReadFile, ImzmlWriteFile


class CreateImzmlPool:
    """Combines randomly sampled spectra from several files into a pool, which can be used for further processing."""

    def __init__(self, source_files: list[ImzmlReadFile], n_spectra_per_file: int, random_seed: int = 0) -> None:
        self._source_files = sorted(source_files, key=lambda x: x.imzml_file)
        self._n_spectra_per_file = n_spectra_per_file
        self._random_seed = random_seed

    def write_pool(self, write_file: ImzmlWriteFile) -> None:
        self._create_output_directory(write_file=write_file)
        self._write_metadata(write_file=write_file)
        self._write_imzml_file(write_file=write_file)

    @cached_property
    def pool_source_df(self) -> pd.DataFrame:
        spectra_indices = self._sample_spectra_indices()
        collect = defaultdict(list)
        for file_id, (imzml_file, indices) in enumerate(zip(self._source_files, spectra_indices)):
            collect["file_id"].append(file_id)
            collect["abs_path"].append(os.path.abspath(imzml_file.imzml_file))
            collect["rel_path"].append(os.path.basename(imzml_file.imzml_file))
            collect["source_spectrum_id"].append(indices)
            collect["n_spectra"].append(imzml_file.n_spectra)
        return pd.DataFrame(collect)

    @cached_property
    def pool_content_df(self) -> pd.DataFrame:
        collect = defaultdict(list)
        offset = 0
        for _, row in self.pool_source_df.iterrows():
            source_spectrum_ids = row["source_spectrum_id"]
            collect["file_id"].extend([row["file_id"]] * len(source_spectrum_ids))
            collect["abs_path"].extend([row["abs_path"]] * len(source_spectrum_ids))
            collect["rel_path"].extend([row["rel_path"]] * len(source_spectrum_ids))
            collect["pool_spectrum_id"].extend(np.arange(offset, offset + len(source_spectrum_ids)))
            collect["source_spectrum_id"].extend(source_spectrum_ids)
            offset += len(source_spectrum_ids)
        return pd.DataFrame(collect)

    def _create_output_directory(self, write_file: ImzmlWriteFile) -> None:
        output_directory = os.path.dirname(write_file.imzml_file)
        os.makedirs(output_directory, exist_ok=True)

    def _write_metadata(self, write_file: ImzmlWriteFile) -> None:
        pool_source_path = os.path.splitext(write_file.imzml_file)[0] + "_source.json"
        pool_content_path = os.path.splitext(write_file.imzml_file)[0] + "_content.json"
        self.pool_source_df.to_json(pool_source_path, index=False, indent=2)
        self.pool_content_df.to_json(pool_content_path, index=False, indent=2)

    def _write_imzml_file(self, write_file: ImzmlWriteFile) -> None:
        """Writes the pooled spectra to the .imzML file."""
        with write_file.writer() as writer:
            for imzml_file in tqdm(self._source_files, desc="Copying file"):
                with imzml_file.reader() as reader:
                    os.path.abspath(imzml_file.imzml_file)
                    spectrum_ids = self.pool_source_df.query("abs_path == @imzml_path").iloc[0]["source_spectrum_id"]
                    writer.copy_spectra(reader, spectrum_ids)

    def _sample_spectra_indices(self) -> list[NDArray[int]]:
        """Returns a list with one element for every file containing the indices of the sampled spectra."""
        rng = np.random.default_rng(self._random_seed)
        per_file_indices = []
        for imzml_file in self._source_files:
            n_spectra = imzml_file.n_spectra
            if n_spectra < self._n_spectra_per_file:
                raise ValueError(
                    f"File {imzml_file.imzml_file} has only {n_spectra} spectra, "
                    f"but {self._n_spectra_per_file} are requested"
                )
            indices = rng.choice(n_spectra, self._n_spectra_per_file, replace=False)
            per_file_indices.append(np.sort(indices))
        return per_file_indices
