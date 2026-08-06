"""A single streaming pass over an imzML, answering the two questions imzy cannot.

**Compression.** imzy reads the ``.ibd`` as raw little-endian floats: ``_read_spectrum`` is
a bare ``np.frombuffer(mz_bytes, dtype=self.mz_precision)``, nothing anywhere in the package
inspects the compression cvParam, and ``byte_offsets`` stores *array* lengths rather than
*encoded* lengths. A compressed file therefore does not fail -- it yields plausible-looking
noise. That is the single most dangerous property of the backend, so it is checked before
any read rather than being left to whatever the numbers look like downstream.

# upstream: no zlib support in imzy; report the silent-corruption behaviour as a bug in
# its own right, independent of the fix. See ROADMAP.md, Phase D gap (1).

**Whether a z coordinate is declared.** imzy's ``xyz_coordinates`` is always ``(n, 3)``,
substituting ``z = 1`` where ``IMS:1000052`` is absent, while the legacy reader returns
``(n, 2)`` for such a file. That difference is not cosmetic: ``reader.coordinates[i]`` is
handed straight to ``writer.add_spectrum`` in several tools, so a backend that invented a z
column would silently turn every 2D acquisition into a 3D one on copy.

Both answers come from one walk, and the walk uses ``iterparse`` rather than
``ParseMetadata`` because acquisitions are routinely multi-GB, this runs on open, and the
compression cvParam may sit either on a ``referenceableParamGroup`` near the top of the file
or on an individual ``binaryDataArray`` -- only a full traversal sees both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from xml.etree import ElementTree

if TYPE_CHECKING:
    from pathlib import Path

_NS = "{http://psi.hupo.org/ms/mzml}"
_CV_PARAM = f"{_NS}cvParam"
_SPECTRUM = f"{_NS}spectrum"
_SPECTRUM_LIST = f"{_NS}spectrumList"

#: `IMS:1000052`, the "position z" cvParam.
_POSITION_Z = "IMS:1000052"

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


@dataclass(frozen=True)
class ImzmlScan:
    """What one pass over an imzML found.

    Cheap to pickle, so it travels to worker processes with the read file instead of being
    recomputed in each of them.
    """

    #: Accession of the first unsupported compression cvParam, or None when there is none.
    compression: str | None
    #: True when at least one spectrum declares `IMS:1000052`. Undefined -- and irrelevant --
    #: when `compression` is set, because the scan stops early in that case.
    declares_z: bool

    def raise_if_unsupported_compression(self, path: str | Path) -> None:
        """Raises `UnsupportedCompressionError` if the scanned file is compressed.

        Args:
            path: the scanned file, used for the error message only.
        """
        if self.compression is not None:
            raise UnsupportedCompressionError(
                f"{path} declares {self.compression} ({UNSUPPORTED_COMPRESSION[self.compression]}), which the imzy"
                f" backend cannot decode -- it would read the .ibd as uncompressed floats and return noise. Use the"
                f" default ImzmlReadFile backend, which supports zlib."
            )


def scan_imzml(path: str | Path) -> ImzmlScan:
    """Walks an .imzML once, reporting compression and whether a z coordinate is declared.

    The walk stops at the first unsupported compression accession: the caller raises in that
    case, so nothing further about the file is worth learning.

    Args:
        path: the .imzML file to scan.
    """
    # The `spectrumList` element is where the parser accumulates completed spectra, so it,
    # not the root, is what has to be emptied to keep memory flat on a large acquisition.
    spectrum_list = None
    declares_z = False
    for event, element in ElementTree.iterparse(path, events=("start", "end")):
        if event == "start":
            if element.tag == _SPECTRUM_LIST:
                spectrum_list = element
        elif element.tag == _CV_PARAM:
            accession = element.get("accession")
            if accession in UNSUPPORTED_COMPRESSION:
                return ImzmlScan(compression=accession, declares_z=declares_z)
            if accession == _POSITION_Z:
                declares_z = True
        elif element.tag == _SPECTRUM and spectrum_list is not None:
            spectrum_list.clear()
    return ImzmlScan(compression=None, declares_z=declares_z)


def raise_if_unsupported_compression(path: str | Path) -> None:
    """Raises `UnsupportedCompressionError` if the file's binary arrays are compressed.

    Args:
        path: the .imzML file to scan.
    """
    scan_imzml(path).raise_if_unsupported_compression(path)
