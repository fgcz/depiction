from pydantic import BaseModel

from depiction.tools.correct_baseline.config import BaselineCorrectionConfig
from depiction.tools.filter_peaks.config import FilterPeaksConfig
from depiction.tools.pick_peaks.config import PickPeaksConfig


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
