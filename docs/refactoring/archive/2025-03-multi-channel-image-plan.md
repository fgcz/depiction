# Design Document: MultiChannelImage Refactoring

## Overview

This document outlines a refactoring proposal to split the current `MultiChannelImage` class into two separate implementations:

1. `MultiChannelImage` - A simpler class without masking functionality
2. `MultiChannelMaskedImage` - A class that includes explicit foreground/background masking

## Motivation

The current implementation combines both masked and unmasked functionality in a single class, which leads to:
- Increased complexity in the API
- Overhead when masking isn't needed
- Less clear purpose for the class

By separating these concerns, we can achieve:
- Better separation of responsibilities
- More efficient memory usage when masks aren't needed
- Clearer, more focused APIs for each use case

## Implementation Decisions

The following decisions have been made for this refactoring:

1. **Class Relationship**: Use composition rather than inheritance
2. **Terminology**: Rename `is_foreground` to `mask` throughout the codebase
3. **Backward Compatibility**: Focus on maintaining conceptual functionality rather than strict API compatibility
4. **Performance**: Maintain current performance characteristics

## Class Structure

```
MultiChannelImage - Basic functionality without masking
MultiChannelMaskedImage - Complete functionality with masking
```

These will be independent classes that share common utilities but do not inherit from each other.

## Implementation Details

### 1. Shared Utilities Module

Create a module with pure functions that both classes can use:

```python
# image_utils.py
import numpy as np
import xarray
from xarray import DataArray

def validate_data_dimensions(data: DataArray) -> None:
    # Common validation logic
    pass

def validate_channel_names(data: DataArray) -> None:
    # Common validation logic
    pass

def compute_mask(data: DataArray, bg_value: float = 0.0) -> DataArray:
    """Computes a mask from the data."""
    if np.isnan(bg_value):
        return ~data.isnull().all(dim="c")
    else:
        return (data != bg_value).any(dim="c")
```

### 2. MultiChannelImage Class

```python
class MultiChannelImage:
    """Represents a multi-channel 2D image, internally backed by a `xarray.DataArray`."""

    def __init__(self, data: DataArray) -> None:
        # Assign the data
        self._data = data.transpose("y", "x", "c").drop_attrs()
        
        # Validate the input
        image_utils.validate_data_dimensions(self._data)
        image_utils.validate_channel_names(self._data)
    
    # Basic properties and methods - no masking functionality
    
    @classmethod
    def from_spatial(cls, data: DataArray) -> MultiChannelImage:
        """Creates a MultiChannelImage from a spatial data array."""
        return cls(data=data)

    @classmethod
    def from_flat(cls, values: DataArray, coordinates: DataArray, 
                  channel_names: list[str] | bool = False) -> MultiChannelImage:
        """Creates a MultiChannelImage from flat data."""
        # Implementation that doesn't need mask
        pass
```

### 3. MultiChannelMaskedImage Class

```python
class MultiChannelMaskedImage:
    """Represents a multi-channel 2D image with masking functionality."""

    def __init__(self, data: DataArray, mask: DataArray, mask_label: str = "mask") -> None:
        # Assign the data
        self._data = data.transpose("y", "x", "c").drop_attrs()
        self._mask = mask.transpose("y", "x").drop_vars("c", errors="ignore").drop_attrs()
        self._mask_label = mask_label
        
        # Validate the input
        image_utils.validate_data_dimensions(self._data)
        image_utils.validate_channel_names(self._data)
        self._validate_mask()
    
    def _validate_mask(self) -> None:
        # Mask validation logic
        pass
    
    # All properties and methods including masking functionality
    
    @classmethod
    def from_spatial(cls, data: DataArray, bg_value: float = 0, 
                    mask_label: str = "mask") -> MultiChannelMaskedImage:
        """Creates a MultiChannelMaskedImage from spatial data."""
        mask = image_utils.compute_mask(data=data, bg_value=bg_value)
        return cls(data=data, mask=mask, mask_label=mask_label)
```

### 4. MaskChannel Class (Renamed from AlphaChannel)

Rename the existing `AlphaChannel` class to `MaskChannel` for terminology consistency:

```python
class MaskChannel:
    """Implements logic to stack a mask channel on top of an arbitrary channel image and split it off again."""
    
    def __init__(self, label: str) -> None:
        self._mask_label = label

    def stack(self, data_array: DataArray, mask_array: DataArray) -> DataArray:
        """Stacks the mask channel on top of the data array."""
        return xarray.concat(
            [data_array, mask_array.expand_dims("c", axis=-1).assign_coords(c=[self._mask_label])], dim="c"
        )

    def split(self, combined: DataArray) -> tuple[DataArray, DataArray]:
        """Splits the mask channel off the combined array."""
        data_array = combined.drop_sel(c=self._mask_label)
        mask_array = combined.sel(c=self._mask_label).drop_vars("c").astype(bool)
        return data_array, mask_array
```

### 5. HDF5ImageFormat Class Updates

Modify the `HDF5ImageFormat` class to work with both image types:

```python
class Hdf5ImageFormat:
    """Implements a HDF5 based persistence format for images."""
    
    def __init__(self, image: Union[MultiChannelImage, MultiChannelMaskedImage]) -> None:
        self._image = image
        self._has_mask = hasattr(image, '_mask')
        self._mask_channel = MaskChannel(label=image._mask_label if self._has_mask else "mask")
    
    def write_hdf5(self, path: Path, mode: Literal["a", "w"] = "w", group: str | None = None) -> None:
        """Writes the image to a HDF5 file."""
        data_array = self._image.data_spatial
        
        if self._has_mask:
            mask_array = self._image.mask
            combined_array = self._mask_channel.stack(data_array=data_array, mask_array=mask_array)
            combined_array.attrs["mask_label"] = self._image.mask_label
        else:
            # If no mask, create an empty one for consistency
            empty_mask = xarray.zeros_like(data_array.isel(c=0), dtype=bool)
            combined_array = self._mask_channel.stack(data_array=data_array, mask_array=empty_mask)
            combined_array.attrs["has_mask"] = False
        
        combined_array.to_netcdf(path, mode=mode, group=group, format="NETCDF4", engine="netcdf4")
    
    @classmethod
    def read_hdf5(cls, path: Path, group: str | None = None) -> Union[MultiChannelImage, MultiChannelMaskedImage]:
        """Reads an image from a HDF5 file, returning the appropriate type."""
        combined_array = xarray.open_dataarray(path, group=group)
        
        # Determine if the file has a meaningful mask
        has_mask = combined_array.attrs.get("has_mask", True)
        mask_label = combined_array.attrs.get("mask_label", "mask")
        
        # Split the data
        data_array, mask_array = MaskChannel(label=mask_label).split(combined=combined_array)
        
        if has_mask:
            return MultiChannelMaskedImage(data=data_array, mask=mask_array, mask_label=mask_label)
        else:
            return MultiChannelImage(data=data_array)
```

### 6. ImageChannelStats Updates

Modify `ImageChannelStats` to work with both image types:

```python
class ImageChannelStats:
    """Provides statistics for image channels."""
    
    def __init__(self, image: Union[MultiChannelImage, MultiChannelMaskedImage]) -> None:
        self._image = image
        self._has_mask = hasattr(image, '_mask')
    
    def _get_channel_values(self, i_channel: int, drop_missing: bool) -> np.ndarray:
        """Returns the values of the given channel."""
        if self._has_mask:
            data_channel = self._image.data_flat.isel(c=i_channel).values
            if drop_missing:
                data_channel = data_channel[self._image.fg_mask_flat]
        else:
            # For unmasked images, just get the values directly
            data_channel = self._image.data_spatial.isel(c=i_channel).values.flatten()
            if drop_missing:
                data_channel = data_channel[~np.isnan(data_channel)]
        
        return data_channel
    
    # Other methods remain similar, using the above helper
```

## Factory Functions for Cross-Class Operations

Add factory functions for operations that could create either type:

```python
def read_image_from_hdf5(path: Path, group: str | None = None) -> Union[MultiChannelImage, MultiChannelMaskedImage]:
    """Factory function that creates the appropriate image type based on file contents."""
    return Hdf5ImageFormat.read_hdf5(path=path, group=group)

def create_masked_from_unmasked(image: MultiChannelImage, bg_value: float = 0.0) -> MultiChannelMaskedImage:
    """Creates a masked image from an unmasked one."""
    mask = image_utils.compute_mask(image.data_spatial, bg_value=bg_value)
    return MultiChannelMaskedImage(data=image.data_spatial, mask=mask)
```

## Migration Guide for Existing Code

For code that currently uses `MultiChannelImage`:

1. If the code doesn't use masking functionality:
   - Switch to the new `MultiChannelImage` class
   - Remove any mask-related parameters

2. If the code uses masking functionality:
   - Switch to `MultiChannelMaskedImage`
   - Update parameter names from `is_foreground` to `mask`
   - Update attribute/method references from `is_foreground_label` to `mask_label`

3. For code that loads from HDF5:
   - Use the factory function `read_image_from_hdf5`
   - Check the type of the returned object if necessary

## Testing Strategy

1. Create tests for both classes independently
2. Test conversion between the two classes
3. Test persistence compatibility
4. Test performance to ensure no regressions

## Future Enhancements

### Real-Valued Alpha Channel Support

In the future, we should consider adding support for real-valued alpha channels (continuous transparency values between 0 and 1) rather than just boolean masks. This would be useful for:

1. Partial transparency effects
2. Confidence or probability maps
3. Gradual transitions between regions
4. Weighted compositing of multiple images

Implementation considerations:

```python
class MultiChannelAlphaImage:
    """Represents a multi-channel image with a continuous alpha channel."""
    
    def __init__(self, data: DataArray, alpha: DataArray, alpha_label: str = "alpha") -> None:
        self._data = data.transpose("y", "x", "c").drop_attrs()
        self._alpha = alpha.transpose("y", "x").drop_vars("c", errors="ignore").drop_attrs()
        self._alpha_label = alpha_label
        
        # Similar validation as masked image
        # But alpha can be float between 0-1 instead of boolean
```

This would require:
1. A separate persistence mechanism for alpha values
2. Modified statistics calculations to handle weighted contributions
3. Different compositing operations based on alpha values
4. Conversion methods between the different image types

### Consistent Naming Update

With the renaming of `is_foreground` to `mask` and `AlphaChannel` to `MaskChannel`, we need to consistently update terminology throughout the codebase. Here's a comprehensive list of terms to update:

| Current Term | New Term |
|--------------|----------|
| `AlphaChannel` | `MaskChannel` |
| `alpha_channel` | `mask_channel` |
| `_alpha_channel` | `_mask_channel` |
| `_alpha_label` | `_mask_label` |
| `is_foreground` | `mask` |
| `is_foreground_label` | `mask_label` |
| `_is_foreground` | `_mask` |
| `_is_foreground_label` | `_mask_label` |
| `is_fg_array` | `mask_array` |
| `fg_mask` | `mask` |
| `bg_mask` | `inverse_mask` (or directly use `~mask`) |
| `fg_mask_flat` | `mask_flat` |
| `bg_mask_flat` | `inverse_mask_flat` |
| `_assert_foreground_is_boolean` | `_assert_mask_is_boolean` |
| `_assert_data_and_foreground_dimensions` | `_assert_data_and_mask_dimensions` |
| `_assert_data_and_foreground_coords` | `_assert_data_and_mask_coords` |
| `_compute_is_foreground` | `_compute_mask` |
| `recompute_is_foreground` | `recompute_mask` |

File paths to update:
- `src/depiction/image/container/alpha_channel.py` → `src/depiction/image/container/mask_channel.py`

We should also search for these terms in:
1. Documentation and comments
2. Test files and fixtures
3. Any import statements
4. CLI commands or user-facing parameters
5. Configuration files