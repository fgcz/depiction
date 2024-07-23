import xarray
from depiction.image.multi_channel_image import MultiChannelImage


def horizontal_concat(
    images: list[MultiChannelImage], *, add_index: bool = False, index_channel: str = "image_index"
) -> MultiChannelImage:
    """Concatenates images horizontally.
    :param images: the images to concatenate
    :param add_index: whether a channel with the index of the image should be added
    :param index_channel: the name of the channel containing the index, if it is added
    """
    if len(images) < 2:
        raise ValueError("At least two images are required for concatenation.")
    if any(image.channel_names != images[0].channel_names for image in images):
        raise ValueError("All images must have the same channel names.")
    ymax = max(image.data_spatial.y.values.max() for image in images)

    # shift x coordinates iteratively
    xoffset = 0
    concat = []
    bg_value = images[0].bg_value
    for i_image, image in enumerate(images):
        data = image.data_spatial
        data = data.pad(y=(0, ymax - data.y.values.max()), constant_values=bg_value)
        x_extent = data.x.values.max() - data.x.values.min() + 1
        data_shifted = data.assign_coords(x=data.x - data.x.values.min() + xoffset)
        if add_index:
            data_index = xarray.full_like(data_shifted.isel(c=[0]), i_image).assign_coords({"c": [index_channel]})
            data_shifted = xarray.concat([data_shifted, data_index], dim="c")
        concat.append(data_shifted)
        xoffset += x_extent
    data = xarray.concat(concat, dim="x")
    return MultiChannelImage(data)
