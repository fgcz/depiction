from pydantic import BaseModel

from depiction_tools.process_spectra.correct_baseline.config import BaselineCorrectionConfig
from depiction_tools.process_spectra.filter_peaks.config import FilterPeaksConfig
from depiction_tools.process_spectra.pick_peaks.config import PickPeaksConfig


class ProcessSpectraStepPickPeaks(BaseModel):
    pick: PickPeaksConfig


class ProcessSpectraStepRemoveBaseline(BaseModel):
    baseline: BaselineCorrectionConfig


class ProcessSpectraStepFilterPeaks(BaseModel):
    filter: FilterPeaksConfig


ProcessSpectraStep = ProcessSpectraStepFilterPeaks | ProcessSpectraStepPickPeaks | ProcessSpectraStepRemoveBaseline


class ProcessSpectraConfig(BaseModel):
    steps: list[ProcessSpectraStep]
    n_jobs: int = 10
