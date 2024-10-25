# MSI Data

## Mass spectrometry imaging data

### Reading data

We have a `GenericReadFile` protocol, which encodes a container file handle, from which we can obtain `GenericReader` instances,
which perform the actual reading of the data.

In general the idea is that creating the file should be quick, whereas additional parsing might be necessary to create the reader instance.

```{eval-rst}
.. autoclass:: depiction.persistence.types.GenericReadFile
    :members:
.. autoclass:: depiction.persistence.types.GenericReader
    :members:
```

### Writing data

```{eval-rst}
.. autoclass:: depiction.persistence.types.GenericWriteFile
    :members:
.. autoclass:: depiction.persistence.types.GenericWriter
    :members:
```

### Format: ImzML

Our ImzML functionality, essentially wraps pyImzML.

### Format: RAM
