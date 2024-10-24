import unittest


from depiction.calibration.methods.calibration_method_mcc import CalibrationMethodMassClusterCenterModel


class TestCalibrationMethodMCC(unittest.TestCase):
    def test_get_distances_nearest_when_invalid(self) -> None:
        mccm = CalibrationMethodMassClusterCenterModel(model_smoothing_activated=False)


if __name__ == "__main__":
    unittest.main()
