import h5py
import numpy as np
import spatialdata


class ExportSpatialdata:
    """
    Exports hdf5 store of labels to spatialdata format, which uses zarr format.
    """

    def to_spatialdata_zarr(self, hdf5_path: str, out_path: str, dtype: np.float32) -> None:
        data = self.to_spatialdata(hdf5_path=hdf5_path)
        data.write(out_path)

    def to_spatialdata(self, hdf5_path: str, dtype=np.float32) -> spatialdata.SpatialData:
        # Load the necessary information
        with h5py.File(hdf5_path, "r") as file:
            images_2d = np.asarray(file["ion_images/images_2d"])
            labels = [self._safe_label(label) for label in file["ion_images/images_2d"].attrs["labels"]]

        # Export the data
        images = {}
        for i_channel, label in enumerate(labels):
            images[label] = spatialdata.models.Image2DModel.parse(images_2d[:, :, [i_channel]], dims=("y", "x", "c"))
        return spatialdata.SpatialData(images=images)

    def _safe_label(self, label: str) -> str:
        return label.replace("/", "_").replace(".", "_")
