# TODO it's unclear where this belongs and should be refactored later
import os

import h5py
import numpy as np

from ionplotter.image.sparse_image_2d import SparseImage2d


class IonvizLabeledOutputs:
    @staticmethod
    def get_labeled_outputs(output_dir):
        h5_file_path = os.path.join(output_dir, "data.hdf5")
        with h5py.File(h5_file_path, "r") as file:
            channel_names = list(file["ion_images/images_2d"].attrs["labels"])
            dense_values = np.asarray(file["ion_images/images_2d"])
            offset = file["ion_images/images_2d"].attrs["offset"]

        dense_values = np.flip(dense_values, axis=0)

        return SparseImage2d.from_dense_array(dense_values=dense_values, offset=offset, channel_names=channel_names)


class CompareOutputs:
    def __init__(self, outputs: dict[str, SparseImage2d]) -> None:
        self._outputs = outputs

    def check_channel_names(self) -> None:
        channel_names = [set(output.channel_names) for output in self._outputs.values()]
        if any(channel_names[0] != channel_name for channel_name in channel_names):
            raise ValueError("Channel names are not the same across all outputs", channel_names)

    def get_combined_image(self) -> SparseImage2d:
        renamed_images = [
            image.with_channel_names([f"{key}_{channel_name}" for channel_name in image.channel_names])
            for key, image in self._outputs.items()
        ]
        return SparseImage2d.combine_in_parallel(images=renamed_images)
