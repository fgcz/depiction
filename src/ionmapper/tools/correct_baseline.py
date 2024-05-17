from __future__ import annotations
import enum
from pathlib import Path

import numpy as np
import typer

from ionmapper.evaluate_local_medians_baseline import EvaluateLocalMediansBaseline
from ionmapper.evaluate_tophat_baseline import EvaluateTophatBaseline
from ionmapper.parallel_ops.parallel_config import ParallelConfig
from ionmapper.parallel_ops.write_spectra_parallel import WriteSpectraParallel
from ionmapper.persistence import (
    ImzmlReadFile,
    ImzmlWriteFile,
    ImzmlWriter,
    ImzmlReader,
)
from numpy.typing import NDArray


class BaselineVariants(str, enum.Enum):
    tophat = "tophat"
    loc_medians = "loc_medians"


class CorrectBaseline:
    """Implements baseline correction for imzml files."""

    def __init__(self, parallel_config: ParallelConfig, baseline_correction) -> None:
        self._parallel_config = parallel_config
        self._baseline_correction = baseline_correction

    @classmethod
    def from_variant(
        cls, parallel_config: ParallelConfig, variant: BaselineVariants = BaselineVariants.tophat
    ) -> CorrectBaseline:
        return cls(parallel_config=parallel_config, variant=variant)

    def evaluate_file(self, read_file: ImzmlReadFile, write_file: ImzmlWriteFile):
        """Evaluates the baseline correction for ``read_file`` and writes the results to ``write_file``."""
        parallel = WriteSpectraParallel.from_config(self._parallel_config)
        parallel.map_chunked_to_file(
            read_file=read_file,
            write_file=write_file,
            operation=self._operation,
            bind_args=dict(baseline_correction=self._baseline_correction)
        )

    def evaluate_spectrum(self, mz_arr: NDArray[float], int_arr: NDArray[float]) -> NDArray[float]:
        try:
            return self._baseline_correction.subtract_baseline(mz_arr, int_arr)
        except np.linalg.LinAlgError:
            # TODO this is a temporary fix, FABC specific hopefully
            print("LinAlgError in baseline correction, returning original spectrum.")
            return int_arr

    @classmethod
    def _operation(
        cls,
        reader: ImzmlReader,
        spectra_ids: list[int],
        writer: ImzmlWriter,
        baseline_correction,
    ):
        correct = cls(
            parallel_config=ParallelConfig.no_parallelism(),
            baseline_correction=baseline_correction,
        )
        for spectrum_index in spectra_ids:
            mz_arr, int_arr = reader.get_spectrum(spectrum_index)
            int_arr = correct.evaluate_spectrum(mz_arr, int_arr)
            writer.add_spectrum(mz_arr, int_arr, coordinates=reader.coordinates[spectrum_index])

    @staticmethod
    def _get_baseline_correction(variant: BaselineVariants):
        if variant == BaselineVariants.tophat:
            return EvaluateTophatBaseline(window_size=5000, window_unit="ppm")
        elif variant == BaselineVariants.loc_medians:
            return EvaluateLocalMediansBaseline(window_size=5000, window_unit="ppm")
        else:
            raise ValueError(f"Unknown baseline variant: {variant}")


def main(
    input_imzml: Path,
    output_imzml: Path,
    n_jobs: int = 20,
    baseline_variant: BaselineVariants = BaselineVariants.tophat,
):
    """Corrects the baseline of `input_imzml` and writes the results to `output_imzml`."""
    parallel_config = ParallelConfig(n_jobs=n_jobs, task_size=None)
    input_file = ImzmlReadFile(input_imzml)
    output_file = ImzmlWriteFile(output_imzml, imzml_mode=input_file.imzml_mode)
    output_imzml.parent.mkdir(exist_ok=True, parents=True)
    correct_baseline = CorrectBaseline.from_variant(parallel_config=parallel_config, variant=baseline_variant)
    correct_baseline.evaluate_file(input_file, output_file)


if __name__ == "__main__":
    typer.run(main)
