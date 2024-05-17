
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from ionmapper.calibration.models import GenericModel
from ionmapper.image.sparse_image_2d import SparseImage2d


class VisualizeMassShiftMap:
    def __init__(self, *, models: list[GenericModel], coordinates: NDArray[int]) -> None:
        self._models = models
        self._coordinates = coordinates

    def get_correction_image(self, test_masses: NDArray[float], unit: str = "m/z") -> SparseImage2d:
        """Returns a SparseImage2d with the correction values for the given test masses.
        Output channels will be named "test mass <mass>".
        :param test_masses: The test masses for which the correction values should be computed.
        :param unit: The unit of the test masses. Either "m/z" or "ppm".
        :return: A SparseImage2d with the correction values for the given test masses.
        """

        def compute_error(model, test_mass):
            if unit == "m/z":
                return model.predict(test_mass)
            elif unit == "ppm":
                return model.predict(test_mass) / test_mass * 1e6
            else:
                raise ValueError(f"Unknown {unit=}")

        correction_values = np.concatenate(
            np.asarray(
                [
                    [compute_error(model=model, test_mass=test_mass) for model in self._models]
                    for test_mass in test_masses
                ]
            ),
            axis=1,
        )
        return SparseImage2d(
            values=correction_values,
            coordinates=self._coordinates,
            channel_names=[f"test mass {mass:.2f}" for mass in test_masses],
        )

    def plot_test_mass_maps_and_histograms(
        self,
        test_masses: NDArray[float],
        same_scale: bool,
        n_bins: int = 50,
        scale_percentile: float = 100.0,
        unit: str = "m/z",
    ):
        """Plots a grid of test mass error maps and the histograms of these errors.
        :param test_masses: The test masses for which the correction values should be computed.
        :param same_scale: If True, all map and histogram intensities will be scaled equally. Otherwise, for each test
            mass the scale will be determined individually.
        :param n_bins: The number of bins for the histograms.
        :param scale_percentile: The percentile of the absolute values of the correction values that will be used to
            determine the scaling.
        :param unit: The unit of the test masses. Either "m/z" or "ppm".
        """
        correction_image = self.get_correction_image(test_masses=test_masses, unit=unit)
        n_masses = len(test_masses)

        fig, axs = plt.subplots(n_masses, 2, figsize=(n_masses * 6, 12), sharex="col" if same_scale else False)
        hist_bins = np.histogram(correction_image.sparse_values.ravel(), bins=n_bins)[1] if same_scale else n_bins

        for i_mass, test_mass in enumerate(test_masses):
            self._plot_row(
                ax_map=axs[i_mass, 0],
                ax_hist=axs[i_mass, 1],
                correction_image=correction_image.retain_channels(channel_indices=[i_mass]),
                hist_bins=hist_bins,
                same_scale=same_scale,
                scale_percentile=scale_percentile,
                unit=unit,
            )

        fig.tight_layout()
        return fig, axs

    def _plot_row(
        self,
        ax_map: plt.Axes,
        ax_hist: plt.Axes,
        correction_image: SparseImage2d,
        hist_bins: NDArray[float] | int,
        same_scale: bool,
        scale_percentile: float,
        unit: str,
    ) -> None:
        # get the data
        values = correction_image.sparse_values.ravel()
        name = correction_image.channel_names[0]

        # scaling value
        if same_scale:
            abs_perc = np.percentile(np.abs(correction_image.sparse_values.ravel()), scale_percentile)
        else:
            abs_perc = np.percentile(np.abs(values), scale_percentile)

        # plot the map
        correction_image.plot_single_channel_image(
            i_channel=0, ax=ax_map, cmap="coolwarm", vmin=-abs_perc, vmax=abs_perc
        )
        ax_map.set_title(f"Error for {name}")
        ax_map.axis("off")

        # plot the histogram
        ax_hist.hist(values, bins=hist_bins)
        ax_hist.set_xlabel(f"Correction [{unit}]")
        ax_hist.set_ylabel("Count")
        ax_hist.set_title(f"Error distribution for {name}")
        ax_hist.grid(True)

        # center histogram at zero
        abs_max = np.max(np.abs(correction_image.sparse_values.ravel())) if same_scale else np.max(np.abs(values))
        ax_hist.set_xlim(-abs_max, abs_max)
