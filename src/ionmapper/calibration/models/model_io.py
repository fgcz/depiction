from ionmapper.calibration.models.generic_model import GenericModel
import pickle


class ModelIO:
    """Currently provides a thin wrapper around pickle for quick and dirty saving and loading of models.
    In the future it would make sense to make the data a bit more long term usable."""

    @staticmethod
    def save_models_pickle(models: list[GenericModel], file_path: str, file_mode="wb"):
        with open(file_path, file_mode) as file:
            pickle.dump(models, file)

    @staticmethod
    def load_models_pickle(file_path: str) -> list[GenericModel]:
        with open(file_path, "rb") as file:
            return pickle.load(file)
