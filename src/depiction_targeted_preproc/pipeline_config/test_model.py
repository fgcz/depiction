import unittest
from pathlib import Path

import yaml
from pydantic import BaseModel
from depiction_targeted_preproc.pipeline_config.model import (
    BaselineAdjustment,
    BaselineAdjustmentTophat,
    PeakPicker,
    PeakPickerBasicInterpolated,
    PipelineParameters,
)


class _ModelBaselineAdjustment(BaseModel):
    x: BaselineAdjustment


class _ModelPeakPicker(BaseModel):
    x: PeakPicker


class TestModel(unittest.TestCase):
    def test_parse_baseline_adjustment_none(self):
        baseline_none = _ModelBaselineAdjustment.model_validate({"x": None})
        self.assertIsNone(baseline_none.x)

    def test_parse_baseline_adjustment_tophat(self):
        baseline_tophat = _ModelBaselineAdjustment.model_validate(
            {"x": {"baseline_type": "Tophat", "window_size": 10, "window_unit": "ppm"}}
        )
        self.assertIsInstance(baseline_tophat.x, BaselineAdjustmentTophat)
        self.assertEqual(baseline_tophat.x.window_size, 10)
        self.assertEqual(baseline_tophat.x.window_unit, "ppm")

    def test_parse_peak_picker_none(self):
        peak_picker = _ModelPeakPicker.model_validate({"x": None})
        self.assertIsNone(peak_picker.x)

    def test_parse_peak_picker_basic_interpolated(self):
        peak_picker = _ModelPeakPicker.model_validate(
            {"x": {"peak_picker_type": "BasicInterpolated", "min_prominence": 10}}
        )
        self.assertIsInstance(peak_picker.x, PeakPickerBasicInterpolated)
        self.assertEqual(peak_picker.x.min_prominence, 10)

    def test_parse_default_pipeline_parameters(self):
        path = Path(__file__).parent / "default.yml"
        pipeline_parameters = PipelineParameters.model_validate(yaml.safe_load(path.read_text()))
        print(pipeline_parameters)


if __name__ == "__main__":
    unittest.main()
