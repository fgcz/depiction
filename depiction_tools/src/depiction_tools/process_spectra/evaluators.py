from functools import cached_property
from numpy.typing import NDArray
from typing import Protocol

from depiction_io.parallel_ops import ParallelConfig
from depiction.spectrum.peak_filtering import PeakFilteringType
from depiction_tools.process_spectra.correct_baseline.config import BaselineCorrectionConfig
from depiction_tools.process_spectra.correct_baseline.correct_baseline import CorrectBaseline
from depiction_tools.process_spectra.filter_peaks.config import FilterPeaksConfig
from depiction_tools.process_spectra.filter_peaks.filter_peaks import get_peak_filter
from depiction_tools.process_spectra.pick_peaks.config import PickPeaksConfig
from depiction_tools.process_spectra.pick_peaks.pick_peaks import get_peak_picker_from_config
from depiction_tools.process_spectra.config import (
    ProcessSpectraStepPickPeaks,
    ProcessSpectraStepRemoveBaseline,
    ProcessSpectraStepFilterPeaks,
    ProcessSpectraConfig,
)
import numpy as np


class Evaluator(Protocol):
    def evaluate(
        self, mz_arr: NDArray[np.float64], int_arr: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        raise NotImplementedError


def get_evaluator(step_config) -> Evaluator:
    match step_config:
        case ProcessSpectraStepPickPeaks(pick=pick_peaks_config):
            return EvaluatePickPeaks(config=pick_peaks_config)
        case ProcessSpectraStepRemoveBaseline(baseline=baseline_config):
            return EvaluateRemoveBaseline(config=baseline_config)
        case ProcessSpectraStepFilterPeaks(filter=filter_peaks_config):
            return EvaluateFilterPeaks(config=filter_peaks_config)
        case _:
            raise ValueError(f"Unsupported step config: {step_config}")


def get_combined_evaluator(config: ProcessSpectraConfig) -> Evaluator:
    evaluators = [get_evaluator(step_config) for step_config in config.steps]
    return CombinedEvaluator(evaluators)


class CombinedEvaluator(Evaluator):
    def __init__(self, evaluators) -> None:
        self._evaluators = evaluators

    def evaluate(self, mz_arr, int_arr):
        for evaluator in self._evaluators:
            mz_arr, int_arr = evaluator.evaluate(mz_arr, int_arr)
        return mz_arr, int_arr


class EvaluatePickPeaks(Evaluator):
    def __init__(self, config: PickPeaksConfig) -> None:
        self._config = config

    @cached_property
    def _picker(self):
        return get_peak_picker_from_config(self._config)

    def evaluate(self, mz_arr, int_arr):
        return self._picker.pick_peaks(mz_arr, int_arr)


class EvaluateRemoveBaseline(Evaluator):
    def __init__(self, config: BaselineCorrectionConfig) -> None:
        self._config = config

    @cached_property
    def _correct_baseline(self) -> CorrectBaseline:
        return CorrectBaseline.from_variant(
            parallel_config=ParallelConfig.no_parallelism(),
            variant=self._config.baseline_variant,
            window_size=self._config.window_size,
            window_unit=self._config.window_unit,
        )

    def evaluate(self, mz_arr, int_arr):
        int_arr_new = self._correct_baseline.evaluate_spectrum(mz_arr, int_arr)
        return mz_arr, int_arr_new


class EvaluateFilterPeaks(Evaluator):
    def __init__(self, config: FilterPeaksConfig) -> None:
        self._config = config

    @cached_property
    def _filter(self) -> PeakFilteringType:
        return get_peak_filter(self._config)

    def evaluate(self, mz_arr, int_arr):
        # TODO this is going to be important, how to handle this, i think we really
        #      need to remove the spectrum_mz_arr, spectrum_int_arr; but then it will
        #      not be possible to implement some things anymore (e.g. relative to total TIC)
        #       unless filters can request these info somehow
        return self._filter.filter_peaks(mz_arr, int_arr, mz_arr, int_arr)
