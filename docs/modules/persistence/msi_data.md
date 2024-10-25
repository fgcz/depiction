## Mass spectrometry imaging data

### Reading data

We have a `GenericReadFile` protocol, which encodes a container file handle, from which we can obtain `GenericReader` instances,
which perform the actual reading of the data.

In general the idea is that creating the file should be quick, whereas additional parsing might be necessary to create the reader instance.

```{eval-rst}
.. autoclass:: depiction.persistence.types.GenericReadFile
    :members:
    :show-inheritance:
.. autoclass:: depiction.persistence.types.GenericReader
    :members:
    :show-inheritance:
```

### Writing data

```{eval-rst}
.. autoclass:: depiction.persistence.types.GenericWriteFile
    :members:
    :show-inheritance:
.. autoclass:: depiction.persistence.types.GenericWriter
    :members:
    :show-inheritance:
```

### Format: ImzML

Currently, we parse imzML ourselves with a simple etree-based parser, whereas writing is performed by pyImzML.
This might be changed under the hood in the future.

```{eval-rst}
.. autoclass:: depiction.persistence.ImzmlModeEnum
    :members:
```

### Format: RAM
