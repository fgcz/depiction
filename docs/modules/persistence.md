The persistence module contains functionality to read and write mass spectrometry imaging (MSI) data from and to the file system.
To keep the rest of depiction as agnostic of the representation as possible, an abstraction is introduced that allows changing the underlying storage without affecting the rest of the code.
In particular, we can parallelize code with the `parallelization` module and use RAM based storage when needed.

## General abstraction

### Reading data

We have a `GenericReadFile` protocol, which encodes a container file handle, from which we can obtain `GenericReader` instances,
which perform the actual reading of the data.

In general the idea is that creating the file should be quick, whereas additional parsing might be necessary to create the reader instance.

::: depiction.persistence.types.GenericReadFile
    options:
        annotations_path: source
        show_category_heading: yes
        show_root_heading: yes
        show_symbol_type_heading: yes
        show_source: no
        members_order: alphabetical
        heading_level: 4

::: depiction.persistence.types.GenericReader
    options:
        annotations_path: source
        show_category_heading: yes
        show_root_heading: yes
        show_symbol_type_heading: yes
        show_source: no
        members_order: alphabetical
        heading_level: 4

## ImzML

Our ImzML functionality, essentially wraps pyImzML.

## Ram
