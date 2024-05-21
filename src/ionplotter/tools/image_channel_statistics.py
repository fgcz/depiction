import xarray
import polars as pl

# TODO figure out ideal package for this class
# TODO add spatial statistics as well in the future


class ImageChannelStatistics:
    """Computes statistics for the channels of a multichannel image."""

    @classmethod
    def compute_for_variants(cls, images: xarray.DataArray, variant_dim: str = "variant") -> pl.DataFrame:
        collect = []
        for variant in images.coords[variant_dim]:
            image = images.sel({variant_dim: variant})
            statistics = cls.compute_xarray(image).with_columns(variant=pl.lit(variant))
            collect.append(statistics)
        return pl.concat(collect)

    @staticmethod
    def compute_xarray(image: xarray.DataArray) -> pl.DataFrame:
        """Compute statistics for the channels of a multichannel image."""
        # convert to expected shape
        image = image.transpose("c", "y", "x")

        # compute statistics
        means = image.mean(dim=["y", "x"])
        stds = image.std(dim=["y", "x"])
        sums = image.sum(dim=["y", "x"])
        p25 = image.quantile(0.25, dim=["y", "x"])
        p50 = image.quantile(0.50, dim=["y", "x"])
        p75 = image.quantile(0.75, dim=["y", "x"])
        mins = image.min(dim=["y", "x"])
        maxs = image.max(dim=["y", "x"])

        # combine statistics
        return pl.DataFrame(
            {
                "channel": image.coords["c"].values,
                "int_mean": means.values,
                "int_std": stds.values,
                "int_sum": sums.values,
                "int_p25": p25.values,
                "int_p50": p50.values,
                "int_p75": p75.values,
                "int_min": mins.values,
                "int_max": maxs.values,
            }
        )
