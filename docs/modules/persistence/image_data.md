## Image data

We use [bioio](https://github.com/bioio-devs/bioio) for image data persistence.
This will allow us to improve interoperability in the future if necessary, nevertheless we currently
have some small helpers for the most important formats we support.

### Format: OME-TIFF

The following methods are available to read and write `MultiChannelImage` objects to and from OME-TIFF files.

```{eval-rst}
.. automethod:: depiction_io.format_ome_tiff.OmeTiff.read_image
.. automethod:: depiction_io.format_ome_tiff.OmeTiff.write_image
```

### Format: OME-NGFF

To be implemented, should be trivial now that we use bioio.

### Format: NetCDF4
