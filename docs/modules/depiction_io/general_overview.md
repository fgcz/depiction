# General overview

`depiction_io` contains functionality to read and write mass spectrometry imaging (MSI) data from
and to the file system.
To keep the rest of depiction as agnostic of the representation as possible, an abstraction is
introduced that allows changing the underlying storage without affecting the rest of the code.
In particular, we can parallelize code with `depiction.parallel_ops` for any of our MSI data
persistence implementations.

Image data reading and writing used to live here too, but it takes and returns `MultiChannelImage`
and so lives in `depiction.image` instead; see [Image data](../image/image_data.md).
