# MultiChannelImage class
This class makes our code a bit more standardized on how multi-channel images are handled internally.
We use Xarray for the representation of the data and basically provide a wrapper around it providing reusable blocks of common functionality.

The basic convention is to use the dimensions (c, y, x) for the channel, y and x dimensions, respectively.
The channels have coordinates in the DataArray, describing the channel names.

## Mask-based / Sparse representation

TODO to be described

## Flat representation

TODO To be described, since it is a bit tricky with the whole multi-index support.

## API Reference

::: depiction.image.MultiChannelImage
    options:
        heading_level: 3
