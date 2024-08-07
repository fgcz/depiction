from pydantic import BaseModel, Field
from typing import Literal, Annotated, Union

from depiction.image.multi_channel_image import MultiChannelImage


class FeatureSelectionCV(BaseModel):
    method: Literal["CV"] = "CV"
    n_features: int


class FeatureSelectionIQR(BaseModel):
    method: Literal["IQR"] = "IQR"
    n_features: int


FeatureSelection = Annotated[Union[FeatureSelectionCV, FeatureSelectionIQR], Field(discriminator="method")]


def select_features(feature_selection: FeatureSelection, image: MultiChannelImage) -> MultiChannelImage:
    """Returns a new ``MultiChannelImage`` that is a copy of ``image`` with only the selected features remaining."""
    match feature_selection:
        case FeatureSelectionCV(n_features=n_features):
            features = _select_features_cv(image=image, n_features=n_features)
        case FeatureSelectionIQR(n_features=n_features):
            features = _select_features_iqr(image=image, n_features=n_features)
        case _:
            raise ValueError("Invalid feature selection method.")
    return image.retain_channels(coords=features)


def _select_features_cv(image: MultiChannelImage, n_features: int) -> list[str]:
    cv = image.channel_stats.coefficient_of_variation
    n_channels = len(cv)
    return cv.drop_nulls().sort("cv").tail(min(n_features, n_channels))["c"].to_list()


def _select_features_iqr(image: MultiChannelImage, n_features: int) -> list[str]:
    iqr = image.channel_stats.interquartile_range
    n_channels = len(iqr)
    return iqr.drop_nulls().sort("iqr").tail(min(n_features, n_channels))["c"].to_list()
