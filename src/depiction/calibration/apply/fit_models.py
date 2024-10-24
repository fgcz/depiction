import numpy as np
import xarray
from numpy._typing import NDArray
from xarray import DataArray

from depiction.calibration.calibration_method import CalibrationMethod
from depiction.image import MultiChannelImage
from depiction.parallel_ops import ParallelConfig
from depiction.parallel_ops.parallel_map import ParallelMap


class FitModels:
    def __init__(self, calibration: CalibrationMethod, parallel_config: ParallelConfig) -> None:
        self._calibration = calibration
        self._parallel_config = parallel_config

    def get_image(self, all_features: MultiChannelImage) -> MultiChannelImage:
        result = self.get_all_features(all_features)
        return MultiChannelImage.from_flat(
            result, coordinates=all_features.coordinates_flat, channel_names="c" not in result.coords
        )

    def get_all_features(self, all_features):
        parallel_map = ParallelMap.from_config(self._parallel_config)
        # TODO to be refactored
        all_features_flat = all_features.data_flat
        result = parallel_map(
            operation=self.get_chunk_features,
            tasks=np.array_split(all_features_flat.coords["i"], self._parallel_config.n_jobs),
            reduce_fn=lambda chunks: xarray.concat(chunks, dim="i"),
            bind_kwargs={"all_features": all_features_flat},
        )
        return result

    def get_chunk_features(self, spectra_indices: NDArray[int], all_features: DataArray) -> DataArray:
        collect = []
        for spectrum_id in spectra_indices:
            features = all_features.sel(i=spectrum_id)
            model_coef = self._calibration.fit_spectrum_model(features=features)
            collect.append(model_coef)
        return xarray.concat(collect, dim="i")
