The persistence module contains functionality to read and write mass spectrometry imaging (MSI) data from and to the file system.
To keep the rest of depiction as agnostic of the representation as possible, an abstraction is introduced that allows changing the underlying storage without affecting the rest of the code.
In particular, we can parallelize code with the `parallelization` module and use RAM based storage when needed.

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

## Imaging data

### Format: OME-TIFF

Currently, we provide a small helper to create these files, wrapping existing libraries:

::: depiction.persistence.format_ome_tiff.OmeTiff

### Format: OME-NGFF

To be implemented.
