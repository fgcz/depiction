from typing import Sequence

import numpy as np
from xarray import DataArray

from depiction.image.multi_channel_image import MultiChannelImage


class GenerateLabelImage:
    """Generates a label image (i.e. multi-channel image without noise) that can be used to generate a MSI
    dataset later."""

    def __init__(self, image_height: int, image_width: int, n_labels: int, seed: int = 0) -> None:
        self._layers = []
        self._image_height = image_height
        self._image_width = image_width
        self._n_labels = n_labels
        self._rng = np.random.default_rng(seed)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Returns the shape of the label image (height, width, n_labels)."""
        return self._image_height, self._image_width, self._n_labels

    def sample_circles(
        self, channel_indices: Sequence[int] | None = None, radius_mean: float = 15, radius_std: float = 5
    ) -> list[dict[str, float | int]]:
        if channel_indices is None:
            channel_indices = range(self._n_labels)
        circles = []
        for i_channel in channel_indices:
            center_h = self._rng.uniform(0, self._image_height)
            center_w = self._rng.uniform(0, self._image_width)
            radius = self._rng.normal(radius_mean, radius_std)
            circles.append({"center_h": center_h, "center_w": center_w, "radius": radius, "i_channel": i_channel})
        return circles

    def add_circles(
        self,
        circles: list[dict[str, float]],
    ) -> None:
        label_image = np.zeros(self.shape)
        for circle in circles:
            center_h, center_w = circle["center_h"], circle["center_w"]
            radius = circle["radius"]
            i_label = circle["i_channel"]
            for h in range(self._image_height):
                for w in range(self._image_width):
                    distance = np.sqrt((h - center_h) ** 2 + (w - center_w) ** 2)
                    if distance < radius:
                        label_image[h, w, i_label] = 1

        self._layers.append(label_image)

    def add_stripe_pattern(self, i_channel: int, bandwidth: float, rotation: float = 45.0, phase: float = 0.0) -> None:
        def f(x, y):
            return np.sin(y / bandwidth * 2 * np.pi + np.radians(phase))

        data = np.zeros((self._image_height, self._image_width))
        phi = np.radians(rotation)
        rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
        for i in range(self._image_height):
            for j in range(self._image_width):
                i_rot, j_rot = np.dot(rot, [i, j])
                data[i, j] = (f(i_rot, j_rot) + 1) / 2

        layer = np.zeros(self.shape)
        layer[:, :, i_channel] = data
        self._layers.append(layer)

    def render(self) -> MultiChannelImage:
        blended = np.sum(self._layers, axis=0)
        data = DataArray(blended, dims=("y", "x", "c"), coords={"c": [f"synthetic_{i}" for i in range(self._n_labels)]})
        data.attrs["bg_value"] = 0.0
        return MultiChannelImage(data)
