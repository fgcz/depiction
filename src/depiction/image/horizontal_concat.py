import xarray
from depiction.image.multi_channel_image import MultiChannelImage


def horizontal_concat(images: list[MultiChannelImage]) -> MultiChannelImage:
    """Concatenates images horizontally."""
    if len(images) < 2:
        raise ValueError("At least two images are required for concatenation.")
    if any(image.channel_names != images[0].channel_names for image in images):
        raise ValueError("All images must have the same channel names.")
    ymax = max(image.data_spatial.y.values.max() for image in images)

    # shift x coordinates iteratively
    xoffset = 0
    concat = []
    bg_value = images[0].bg_value
    for image in images:
        data = image.data_spatial
        data = data.pad(y=(0, ymax - data.y.values.max()), constant_values=bg_value)
        x_extent = data.x.values.max() - data.x.values.min() + 1
        data_shifted = data.assign_coords(x=data.x - data.x.values.min() + xoffset)
        concat.append(data_shifted)
        xoffset += x_extent
    data = xarray.concat(concat, dim="x")
    return MultiChannelImage(data)
