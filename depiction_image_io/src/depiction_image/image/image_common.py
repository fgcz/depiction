# image_utils.py
import numpy as np
import xarray
from xarray import DataArray


def validate_data_dimensions(data: DataArray) -> None:
    # Common validation logic
    pass


def validate_channel_names(data: DataArray) -> None:
    if "c" not in data.coords:
        raise ValueError("Data must have a 'c' coordinate for channel names.")
    if data.sizes["c"] > 0 and not isinstance(data.c[0].item(), str):
        raise ValueError(f"Channel names must be strings, but type is: {type(data.c[0].item())}.")


def compute_channel_stats(data: DataArray, mask: DataArray = None) -> dict:
    # Compute statistics, optionally using a mask
    if mask is not None:
        # Use mask for calculations
        pass
    else:
        # Calculate without mask
        pass

    return stats_dict
