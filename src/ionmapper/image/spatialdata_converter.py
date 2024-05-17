from ionmapper.image.sparse_image_2d import SparseImage2d
import spatialdata
from spatialdata import SpatialData


class SpatialDataConverter:
    @classmethod
    def to_spatialdata(cls, sparse_img: SparseImage2d, bg_value: float = 0., image_name: str = "image") -> SpatialData:
        array = sparse_img.to_dense_xarray(bg_value=bg_value)
        image = spatialdata.models.Image2DModel.parse(array, c_coords=array.coords["c"])
        return spatialdata.SpatialData(images={image_name: image})

    # TODO check if needed
    #@classmethod
    #def from_spatialdata(cls, spatial_data:SpatialData, image_name: str="image") ->SparseImage2d:
    #   pass

