from __future__ import annotations

from typing import Literal, Annotated, Protocol

from pydantic import BaseModel, Field

from depiction.image import MultiChannelImage
from depiction_image_ops.smoothing.percentile_filter import PercentileFilter, KernelShape
from depiction_image_ops.smoothing import SpatialSmoothingSparseAware


class GaussianSpatialSmoothingConfig(BaseModel):
    type: Literal["Gaussian"] = "Gaussian"
    kernel_size: int = 27
    kernel_std: float = 10.0


class PercentileFilterSpatialSmoothingConfig(BaseModel):
    type: Literal["PercentileFilter"] = "PercentileFilter"
    kernel_size: int
    percentile: float
    kernel_shape: KernelShape = KernelShape.Square


SpatialSmoothingConfig = Annotated[
    GaussianSpatialSmoothingConfig | PercentileFilterSpatialSmoothingConfig, Field(discriminator="type")
]


class SpatialSmoothingType(Protocol):
    def smooth_image(self, image: MultiChannelImage) -> MultiChannelImage: ...


def get_spatial_smoothing(config: SpatialSmoothingConfig | None) -> SpatialSmoothingType | None:
    match config:
        case None:
            return None
        case GaussianSpatialSmoothingConfig(kernel_size=kernel_size, kernel_std=kernel_std):
            # TODO rename the class to match the config
            return SpatialSmoothingSparseAware(kernel_size=kernel_size, kernel_std=kernel_std)
        case PercentileFilterSpatialSmoothingConfig(
            kernel_size=kernel_size, kernel_shape=kernel_shape, percentile=percentile
        ):
            # TODO rename the class to match the config
            return PercentileFilter(kernel_size=kernel_size, kernel_shape=kernel_shape, percentile=percentile)
        case _:
            raise ValueError(f"Unknown spatial smoothing config {config}")
