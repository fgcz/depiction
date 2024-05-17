from pathlib import Path

import lxml.etree
import numpy as np
import spatialdata
import xarray
from PIL import Image
from geopandas import GeoDataFrame
from numpy.typing import NDArray
from shapely import Polygon

from ionmapper.persistence.format_ome_tiff import OmeTiff
from ionmapper.persistence.pixel_size import PixelSize


def main_tmp(
    mis_file_path: Path, jpg_file_path: Path, channel_img_path: Path, output_msi_cutout: Path, output_slide_cutout: Path
) -> None:
    # create the spatial data object (in the future this could also be done in a separate step/persisted maybe)
    sd_obj = get_sd_object(channel_img_path, jpg_file_path, mis_file_path)

    # add information on how to transform slide scan to msi
    # TODO why does spatialdata not infer this when it's needed (the reason i have to explicitly add it is because otherwise the subsequent computation will fail)
    transform = spatialdata.transformations.get_transformation_between_coordinate_systems(sd_obj, "c_slide", "c_msi")
    sd_obj["slide"].transform["c_msi"] = transform

    # current variant: cut the slide to the msi area, rasterize it and then export both together
    # by doing this the msi data will be the same as imported...
    data_extent = spatialdata.get_extent(sd_obj["msi"], "c_msi")
    min_extent = [data_extent["x"][0], data_extent["y"][0]]
    max_extent = [data_extent["x"][1], data_extent["y"][1]]
    sd_raster = spatialdata.rasterize(
        sd_obj,
        axes=("x", "y"),
        min_coordinate=min_extent,
        max_coordinate=max_extent,
        target_coordinate_system="c_msi",
        target_width=max_extent[0] - min_extent[0],
    )

    # obtain pixel size information (this is also available in the imzml file)
    pixel_size = parse_mis_file_resolution(mis_file_path)

    # get the rasterized images
    msi_cutout = sd_raster["msi_rasterized_images"]
    msi_cutout.attrs["pixel_size"] = pixel_size
    # TODO
    msi_cutout.coords["c"] = sd_obj["msi"].coords["c"]
    slide_cutout = sd_raster["slide_rasterized_images"]
    slide_cutout.attrs["pixel_size"] = pixel_size

    # write the tiff files
    OmeTiff.write(msi_cutout, output_msi_cutout)
    OmeTiff.write(slide_cutout, output_slide_cutout)


def parse_mis_file_areas(mis_file_path: Path) -> dict[str, np.ndarray]:
    tree = lxml.etree.parse(mis_file_path)
    areas = {}
    area_elements = tree.xpath("//Area")
    for area_element in area_elements:
        point_elements = area_element.xpath(".//Point")
        point_coords = [[float(s) for s in point_element.text.split(",")] for point_element in point_elements]
        point_coords = np.asarray(point_coords)
        areas[area_element.attrib["Name"]] = point_coords
    return areas


def parse_mis_file_resolution(mis_file_path: Path) -> PixelSize:
    # TODO this is done quickly for now, but it might be more portable to only use the information in the imzML file
    tree = lxml.etree.parse(mis_file_path)
    raster_elements = tree.xpath("//Raster")
    pixel_sizes = set()
    for raster_element in raster_elements:
        values = raster_element.text.split(",")
        # TODO if these are floats then we get an error, this should be defined properly everywhere (or PixelSize needs unit conversion)
        pixel_sizes.add(PixelSize(int(values[0]), int(values[1]), "micrometer"))
    if len(pixel_sizes) != 1:
        raise NotImplementedError(f"{len(pixel_sizes)=} != 1")
    return pixel_sizes.pop()


def get_bbox(points: NDArray) -> tuple[NDArray, NDArray]:
    # points is a Nx2 array
    return np.min(points, axis=0), np.max(points, axis=0)


def get_sd_object(channel_img_path, jpg_file_path, mis_file_path) -> spatialdata.SpatialData:
    mis_file_areas = parse_mis_file_areas(mis_file_path)
    sd_img_scan = get_sd_img_scan(jpg_file_path)
    sd_img_channels = get_sd_img_channels(channel_img_path, area_coords=list(mis_file_areas.values())[0])
    # sd_img_scan.transformations["c_msi"] =sd_img_channels["transformations"]["c_slide"].inverse()
    shape_cutout = get_sd_shape_cutout(mis_file_areas)
    return spatialdata.SpatialData(
        images={"msi": sd_img_channels, "slide": sd_img_scan},
        shapes={
            # "region":
            "cutout": shape_cutout
        },
    )


def get_sd_shape_cutout(mis_file_areas: dict[str, np.ndarray]):
    if len(mis_file_areas) > 1:
        raise NotImplementedError("Only one area is supported for now")
    area_coords = list(mis_file_areas.values())[0]
    return spatialdata.models.ShapesModel.parse(
        GeoDataFrame({"geometry": [Polygon(area_coords)]}),
        transformations={"c_slide": spatialdata.transformations.transformations.Identity()},
    )


def get_sd_img_channels(channel_img_path: Path, area_coords: NDArray) -> spatialdata.models.Image2DModel:
    channel_img = xarray.open_dataarray(channel_img_path)
    # compute translate and scale from channel_pixels to scan_image's target area bbox
    channel_bbox_min = np.array([channel_img.x.min(), channel_img.y.min()])
    channel_bbox_max = np.array([channel_img.x.max(), channel_img.y.max()])
    target_bbox_min, target_bbox_max = get_bbox(area_coords)
    translate = target_bbox_min - channel_bbox_min
    scale = (target_bbox_max - target_bbox_min) / (channel_bbox_max - channel_bbox_min)
    transform_to_c_slide = spatialdata.transformations.Scale(scale, ("x", "y")).compose_with(
        spatialdata.transformations.Translation(translate, ("x", "y"))
    )
    flipped_img = channel_img.isel(y=slice(None, None, -1))
    # fill nan (TODO)
    flipped_img = flipped_img.copy()
    # TODO?
    #flipped_img = flipped_img.fillna(-1.0)
    # normalize (TODO reconsider)
    # flipped_img = ImageNormalization().normalize_xarray(flipped_img, ImageNormalizationVariant.VEC_NORM)
    return spatialdata.models.Image2DModel.parse(
        flipped_img.astype(np.float32),
        transformations={"c_slide": transform_to_c_slide, "c_msi": spatialdata.transformations.Identity()},
        c_coords=channel_img.coords["c"],
    )


def get_sd_img_scan(jpg_file_path: Path) -> spatialdata.models.Image2DModel:
    scan_image = Image.open(jpg_file_path)
    return spatialdata.models.Image2DModel.parse(
        xarray.DataArray(scan_image, dims=("y", "x", "c")),
        transformations={"c_slide": spatialdata.transformations.transformations.Identity()},
    )


# jpg_file_path = Path("/Users/leo/code/msi/code/msi_targeted_preproc/example/data-raw/64005-B20-47740-G-1209-01.jpg")
# mis_file_path = Path("/Users/leo/code/msi/code/msi_targeted_preproc/example/data-raw/menzha_20231210_S607943_64005-B20-47740-G.mis")
# output_img_path = Path("/Users/leo/code/msi/code/msi_targeted_preproc/example/data-work/menzha_20231210_s607943_64005-b20-47740-g/images_default.hdf5")
