## Image data

We use [bioio](https://github.com/bioio-devs/bioio) for image data persistence.
This will allow us to improve interoperability in the future if necessary, nevertheless we currently
have some small helpers for the most important formats we support.

These live in `depiction.image` rather than in `depiction_io`, because they read and write
`MultiChannelImage` objects -- putting them in the I/O package would make it depend on `depiction`.

### Format: OME-TIFF

The following methods are available to read and write `MultiChannelImage` objects to and from OME-TIFF files.

```{eval-rst}
.. automethod:: depiction.image.ome_tiff.OmeTiff.read_image
.. automethod:: depiction.image.ome_tiff.OmeTiff.write_image
```

### Format: OME-NGFF

To be implemented, should be trivial now that we use bioio.

### Format: HDF5 / NetCDF4

`MultiChannelImage` is persisted to HDF5 using the NetCDF4 conventions that `xarray` writes.
`MultiChannelImage.read_hdf5` and `MultiChannelImage.write_hdf5` delegate to `Hdf5ImageFormat`.

```{eval-rst}
.. autoclass:: depiction.image.hdf5_image_format.Hdf5ImageFormat
    :members:
```
