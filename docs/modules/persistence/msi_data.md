# MSI Data

## Mass spectrometry imaging data

### Reading data

We have a `GenericReadFile` protocol, which encodes a container file handle, from which we can obtain `GenericReader` instances,
which perform the actual reading of the data.

In general the idea is that creating the file should be quick, whereas additional parsing might be necessary to create the reader instance.

::: depiction.persistence.types.GenericReadFile
    options:
        heading_level: 4
        members_order: source

::: depiction.persistence.types.GenericReader
    options:
        heading_level: 4
        members_order: source


### Writing data

### Format: ImzML

Our ImzML functionality, essentially wraps pyImzML.

### Format: RAM
