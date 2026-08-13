"""A single streaming pass over an imzML, answering the three questions imzy cannot.

**Compression.** imzy reads the ``.ibd`` as raw little-endian floats: ``_read_spectrum`` is
a bare ``np.frombuffer(mz_bytes, dtype=self.mz_precision)``, nothing anywhere in the package
inspects the compression cvParam, and ``byte_offsets`` stores *array* lengths
(``IMS:1000103``) rather than *encoded* lengths (``IMS:1000104``). A compressed file
therefore does not fail -- it yields plausible-looking noise.

# upstream: no compression support in imzy; report the silent-corruption behaviour as a bug
# in its own right, independent of the fix. See docs/modules/depiction_io/imzy_backend.md, gap (1).

**The encoded lengths a decompressor needs.** zlib is read rather than refused, because it
is what real acquisitions arrive in. The only thing missing to decompress a block is its
encoded length, so this scan collects one, keyed by the block's offset. Keying on the offset
rather than on ``(spectrum, array)`` means the reader never has to work out which
``binaryDataArray`` was the m/z one, and a continuous file's shared m/z block -- pointed at
by every spectrum -- collapses to a single entry for free.

Numpress stays refused: decoding it needs a codec, not a length.

**Whether a z coordinate is declared.** imzy's ``xyz_coordinates`` is always ``(n, 3)``,
substituting ``z = 1`` where ``IMS:1000052`` is absent, while a 2D file has no z at all.
That difference is not cosmetic: ``reader.coordinates[i]`` is handed straight to
``writer.add_spectrum`` in several tools, so a backend that invented a z column would
silently turn every 2D acquisition into a 3D one on copy.

All three answers come from one walk, and the walk uses ``iterparse`` rather than
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
_BINARY_DATA_ARRAY = f"{_NS}binaryDataArray"

#: `IMS:1000052`, the "position z" cvParam.
_POSITION_Z = "IMS:1000052"

#: `IMS:1000102`, the offset of a binary array within the .ibd.
_EXTERNAL_OFFSET = "IMS:1000102"

#: `IMS:1000104`, the length of a binary array *as stored*, i.e. after compression.
_EXTERNAL_ENCODED_LENGTH = "IMS:1000104"

#: `MS:1000574`. The one compression this backend can undo.
ZLIB_COMPRESSION = "MS:1000574"

#: PSI-MS compression accessions that cannot be undone by decompressing a block, so unlike
#: zlib they are refused rather than read. Nothing in this toolchain writes numpress, but it
#: fails the same silent way as zlib did, so it is named rather than left to chance.
UNSUPPORTED_COMPRESSION: dict[str, str] = {
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

    Travels to worker processes with the read file rather than being recomputed in each of
    them. For an uncompressed file that is a couple of fields; for a compressed one
    `encoded_lengths` holds up to two entries per spectrum, so a 100k-pixel acquisition
    pickles roughly 25 MB. That is the deliberate trade: re-walking a multi-GB XML in every
    worker costs far more than shipping the map once.
    """

    #: Accession of the first compression cvParam that cannot be read, or None.
    unsupported_compression: str | None
    #: True when at least one spectrum declares `IMS:1000052`. Undefined -- and irrelevant --
    #: when `unsupported_compression` is set, because the scan stops early in that case.
    declares_z: bool
    #: `IMS:1000102` offset -> `IMS:1000104` encoded length, for a zlib-compressed file.
    #: None when the file is uncompressed, which is also how the reader decides whether to
    #: decompress at all.
    encoded_lengths: dict[int, int] | None = None

    def raise_if_unsupported_compression(self, path: str | Path) -> None:
        """Raises `UnsupportedCompressionError` if the scanned file is compressed unreadably.

        Args:
            path: the scanned file, used for the error message only.
        """
        if self.unsupported_compression is not None:
            raise UnsupportedCompressionError(
                f"{path} declares {self.unsupported_compression}"
                f" ({UNSUPPORTED_COMPRESSION[self.unsupported_compression]}), which the imzy backend cannot decode --"
                f" it would read the .ibd as uncompressed floats and return noise."
            )


def scan_imzml(path: str | Path) -> ImzmlScan:
    """Walks an .imzML once, reporting compression, encoded lengths and whether z is declared.

    The walk stops at the first *unsupported* compression accession: the caller raises in
    that case, so nothing further about the file is worth learning. zlib does not stop it,
    because the rest of the walk is what makes the file readable.

    Args:
        path: the .imzML file to scan.

    Raises:
        UnsupportedCompressionError: the file turns compression on part-way through, which
            leaves no complete set of encoded lengths to hand the reader.
    """
    # The `spectrumList` element is where the parser accumulates completed spectra, so it,
    # not the root, is what has to be emptied to keep memory flat on a large acquisition.
    spectrum_list = None
    declares_z = False
    # Populated only once zlib has been seen, so an uncompressed file -- the common case --
    # never pays for the map. Valid imzML declares compression on a referenceableParamGroup,
    # which the schema puts before the run, so in practice the flag is set before the first
    # binary array is reached; `binary_arrays_skipped` catches the file where it is not.
    encoded_lengths: dict[int, int] | None = None
    binary_arrays_skipped = 0

    for event, element in ElementTree.iterparse(path, events=("start", "end")):
        if event == "start":
            if element.tag == _SPECTRUM_LIST:
                spectrum_list = element
        elif element.tag == _CV_PARAM:
            accession = element.get("accession")
            if accession in UNSUPPORTED_COMPRESSION:
                return ImzmlScan(unsupported_compression=accession, declares_z=declares_z)
            if accession == ZLIB_COMPRESSION and encoded_lengths is None:
                if binary_arrays_skipped:
                    raise UnsupportedCompressionError(
                        f"{path} declares zlib compression only after {binary_arrays_skipped} uncompressed binary"
                        f" array(s). Mixed or late-declared compression is not supported: the scan cannot produce"
                        f" encoded lengths for the arrays it has already passed."
                    )
                encoded_lengths = {}
            if accession == _POSITION_Z:
                declares_z = True
        elif element.tag == _BINARY_DATA_ARRAY:
            if encoded_lengths is None:
                binary_arrays_skipped += 1
            else:
                offset = element.find(f"{_CV_PARAM}[@accession='{_EXTERNAL_OFFSET}']")
                length = element.find(f"{_CV_PARAM}[@accession='{_EXTERNAL_ENCODED_LENGTH}']")
                if offset is None or length is None:
                    raise UnsupportedCompressionError(
                        f"{path} has a compressed binaryDataArray without both {_EXTERNAL_OFFSET} and"
                        f" {_EXTERNAL_ENCODED_LENGTH}, so its stored length is unknown."
                    )
                encoded_lengths[int(offset.attrib["value"])] = int(length.attrib["value"])
        elif element.tag == _SPECTRUM and spectrum_list is not None:
            spectrum_list.clear()

    return ImzmlScan(unsupported_compression=None, declares_z=declares_z, encoded_lengths=encoded_lengths)


def raise_if_unsupported_compression(path: str | Path) -> None:
    """Raises `UnsupportedCompressionError` if the file's binary arrays are compressed
    in a way this backend cannot read.

    Args:
        path: the .imzML file to scan.
    """
    scan_imzml(path).raise_if_unsupported_compression(path)
