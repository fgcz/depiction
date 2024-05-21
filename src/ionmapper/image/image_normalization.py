import enum

import xarray


# TODO experimental code, untested etc
# TODO in principle the interface this method would require is to apply over the pixels of the image


class ImageNormalizationVariant(enum.Enum):
    VEC_NORM = "vec_norm"
    STD = "std"


# TODO maybe rename to ImageFeatureNormalization
class ImageNormalization:
    def normalize_xarray(self, image: xarray.DataArray, variant: ImageNormalizationVariant) -> xarray.DataArray:
        # First, understand the dimensions of the image.
        known_dims = ["y", "x", "c"]
        missing_dims = set(known_dims) - set(image.dims)
        if missing_dims:
            raise ValueError(f"Missing required dimensions: {missing_dims}")
        index_dims = set(image.dims) - set(known_dims)
        if len(index_dims) == 0:
            return self._normalize_single_xarray(image, variant=variant)
        elif len(index_dims) == 1:
            return self._normalize_multiple_xarray(image, index_dim=index_dims.pop(), variant=variant)
        else:
            raise NotImplementedError("Multiple index columns are not supported yet.")

    def _normalize_single_xarray(self, image: xarray.DataArray, variant: ImageNormalizationVariant):
        if variant == ImageNormalizationVariant.VEC_NORM:
            return image / (((image**2).sum(["c"])) ** 0.5)
        elif variant == ImageNormalizationVariant.STD:
            return (image - image.mean("c")) / image.std("c")
        else:
            raise NotImplementedError(f"Unknown variant: {variant}")

    def _normalize_multiple_xarray(self, image: xarray.DataArray, index_dim: str, variant: ImageNormalizationVariant):
        dataset = image.to_dataset(dim=index_dim)
        normalized = dataset.map(
            lambda x: self._normalize_single_xarray(x, variant=variant),
            keep_attrs=True,
        )
        return normalized.to_array(dim=index_dim)
