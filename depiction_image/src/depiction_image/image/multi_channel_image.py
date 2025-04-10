class MultiChannelImage:
    def __init__(self, data: DataArray) -> None:
        self._data = data.transpose("y", "x", "c").drop_attrs()

        # Use shared validation
        validate_data_dimensions(self._data)
        validate_channel_names(self._data)

    def get_channel_stats(self) -> ImageChannelStats:
        """Returns an object providing channel statistics."""
        stats = compute_channel_stats(self._data)
        return ImageChannelStats.from_dict(stats)
