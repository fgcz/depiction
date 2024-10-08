The persistence module contains functionality to read and write mass spectrometry imaging (MSI) data from and to the file system.
To keep the rest of depiction as agnostic of the representation as possible, an abstraction is introduced that allows changing the underlying storage without affecting the rest of the code.
In particular, we can parallelize code with the `parallelization` for any ouf our MSI data persistence implementations.

This module can generally be thought in two parts, corresponding to MSI data reading and writing, and, image data reading and writing.
