from __future__ import annotations

import enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel, PositiveInt, PositiveFloat

from depiction.parallel_ops.parallel_config import ParallelConfig
from depiction.parallel_ops.write_spectra_parallel import WriteSpectraParallel
from depiction.persistence import ImzmlWriteFile, ImzmlWriter, ImzmlReader, ImzmlReadFile
from depiction.spectrum.baseline.local_medians_baseline import LocalMediansBaseline
from depiction.spectrum.baseline.tophat_baseline import TophatBaseline

if TYPE_CHECKING:
    from numpy.typing import NDArray
    from depiction.spectrum.baseline.baseline import Baseline


class BaselineVariants(str, enum.Enum):
    TopHat = "TopHat"
    LocMedians = "LocMedians"


class BaselineCorrectionConfig(BaseModel, use_enum_values=True, validate_default=True):
    n_jobs: PositiveInt | None = None
    baseline_variant: BaselineVariants = BaselineVariants.TopHat
    window_size: PositiveInt | PositiveFloat = 5000.0
    window_unit: Literal["ppm", "index"] = "ppm"


class CorrectBaseline:
    """Implements baseline correction for imzml files."""

    def __init__(self, parallel_config: ParallelConfig, baseline_correction: Baseline) -> None:
        self._parallel_config = parallel_config
        self._baseline_correction = baseline_correction

    @classmethod
    def from_variant(
        cls,
        parallel_config: ParallelConfig,
        variant: BaselineVariants = BaselineVariants.TopHat,
        window_size: int | float = 5000,
        window_unit: Literal["ppm", "index"] = "ppm",
    ) -> CorrectBaseline:
        """Creates an instance of CorrectBaseline with the specified variant."""
        if variant == BaselineVariants.TopHat:
            baseline_correction = TophatBaseline(window_size=window_size, window_unit=window_unit)
        elif variant == BaselineVariants.LocMedians:
            baseline_correction = LocalMediansBaseline(window_size=window_size, window_unit=window_unit)
        else:
            raise ValueError(f"Unknown baseline variant: {variant}")
        return cls(parallel_config=parallel_config, baseline_correction=baseline_correction)

    def evaluate_file(self, read_file: ImzmlReadFile, write_file: ImzmlWriteFile) -> None:
        """Evaluates the baseline correction for ``read_file`` and writes the results to ``write_file``."""
        parallel = WriteSpectraParallel.from_config(self._parallel_config)
        parallel.map_chunked_to_file(
            read_file=read_file,
            write_file=write_file,
            operation=self._operation,
            bind_args=dict(baseline_correction=self._baseline_correction),
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
    ) -> None:
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
        if variant == BaselineVariants.TopHat:
            return TophatBaseline(window_size=5000, window_unit="ppm")
        elif variant == BaselineVariants.LocMedians:
            return LocalMediansBaseline(window_size=5000, window_unit="ppm")
        else:
            raise ValueError(f"Unknown baseline variant: {variant}")


def correct_baseline(config: BaselineCorrectionConfig, input_imzml: Path, output_imzml: Path) -> None:
    """Removes the baseline from the input imzML file and writes the result to the output imzML file."""
    output_imzml.parent.mkdir(parents=True, exist_ok=True)
    input_file = ImzmlReadFile(input_imzml)
    if config.n_jobs is None:
        # TODO define some sane default for None and -1 n_jobs e.g. use all available up to a limit (None) or use all (1-r)
        n_jobs = 10
    else:
        n_jobs = config.n_jobs
    parallel_config = ParallelConfig(n_jobs=n_jobs)
    output_file = ImzmlWriteFile(output_imzml, imzml_mode=input_file.imzml_mode)
    correct_baseline = CorrectBaseline.from_variant(
        parallel_config=parallel_config,
        variant=config.baseline_variant,
        window_size=config.window_size,
        window_unit=config.window_unit,
    )
    correct_baseline.evaluate_file(input_file, output_file)
