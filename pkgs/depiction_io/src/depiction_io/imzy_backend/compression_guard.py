"""Refuses imzML files whose binary arrays are compressed.

imzy reads the ``.ibd`` as raw little-endian floats: ``_read_spectrum`` is a bare
``np.frombuffer(mz_bytes, dtype=self.mz_precision)``, nothing anywhere in the package
inspects the compression cvParam, and ``byte_offsets`` stores *array* lengths rather than
*encoded* lengths. A compressed file therefore does not fail -- it yields plausible-looking
noise. That is the single most dangerous property of the backend, so it is checked before
any read rather than being left to whatever the numbers look like downstream.

# upstream: no zlib support in imzy; report the silent-corruption behaviour as a bug in
# its own right, independent of the fix. See ROADMAP.md, Phase D gap (1).

The scan streams with ``iterparse`` instead of going through ``ParseMetadata``, for two
reasons: acquisitions are routinely multi-GB and the guard runs on every open, and the
compression cvParam may appear either on a ``referenceableParamGroup`` near the top of the
file or on an individual ``binaryDataArray`` -- only a full walk sees both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from xml.etree import ElementTree

if TYPE_CHECKING:
    from pathlib import Path

_NS = "{http://psi.hupo.org/ms/mzml}"
_CV_PARAM = f"{_NS}cvParam"
_SPECTRUM = f"{_NS}spectrum"
_SPECTRUM_LIST = f"{_NS}spectrumList"

#: PSI-MS compression accessions, none of which imzy can decode. Numpress is listed
#: alongside zlib because it fails the same way -- as noise rather than as an error -- even
#: though nothing in this toolchain writes it.
UNSUPPORTED_COMPRESSION: dict[str, str] = {
    "MS:1000574": "zlib compression",
    "MS:1002312": "MS-Numpress linear prediction compression",
    "MS:1002313": "MS-Numpress positive integer compression",
    "MS:1002314": "MS-Numpress short logged float compression",
    "MS:1002746": "MS-Numpress linear prediction compression followed by zlib compression",
    "MS:1002747": "MS-Numpress positive integer compression followed by zlib compression",
    "MS:1002748": "MS-Numpress short logged float compression followed by zlib compression",
}


class UnsupportedCompressionError(NotImplementedError):
    """Raised when a file's binary arrays are compressed in a way the imzy backend cannot read."""


def find_unsupported_compression(path: str | Path) -> str | None:
    """Returns the accession of the first unsupported compression cvParam, or None.

    Args:
        path: the .imzML file to scan.
    """
    # The `spectrumList` element is where the parser accumulates completed spectra, so it,
    # not the root, is what has to be emptied to keep memory flat on a large acquisition.
    spectrum_list = None
    for event, element in ElementTree.iterparse(path, events=("start", "end")):
        if event == "start":
            if element.tag == _SPECTRUM_LIST:
                spectrum_list = element
        elif element.tag == _CV_PARAM:
            accession = element.get("accession")
            if accession in UNSUPPORTED_COMPRESSION:
                return accession
        elif element.tag == _SPECTRUM and spectrum_list is not None:
            spectrum_list.clear()
    return None


def raise_if_unsupported_compression(path: str | Path) -> None:
    """Raises `UnsupportedCompressionError` if the file's binary arrays are compressed.

    Args:
        path: the .imzML file to scan.
    """
    accession = find_unsupported_compression(path)
    if accession is not None:
        raise UnsupportedCompressionError(
            f"{path} declares {accession} ({UNSUPPORTED_COMPRESSION[accession]}), which the imzy backend cannot"
            f" decode -- it would read the .ibd as uncompressed floats and return noise. Use the default"
            f" ImzmlReadFile backend, which supports zlib."
        )
