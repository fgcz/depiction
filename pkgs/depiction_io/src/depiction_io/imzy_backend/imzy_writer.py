"""imzy's imzML writer, with the two behaviours that make a drop-in swap unsafe corrected.

imzy's writer is the better one -- it writes through temporary files and only renames them
into place once the XML is complete, records a SHA-1 of the `.ibd`, deduplicates identical
m/z arrays, and validates what it is handed -- which is why `depiction_io` writes through it
rather than through `pyimzml`. Two of its choices are still wrong for this codebase:

1. **It always writes a z coordinate.** `_normalize_coordinates` appends ``z = 1`` to a 2D
   coordinate and `_add_scan_list` emits `IMS:1000052` unconditionally, where `pyimzml`
   emits it only for a 3-tuple. Left alone, flipping the writer would add a z axis to every
   2D acquisition the pipeline has ever produced, and since `reader.coordinates[i]` is
   handed straight back to `add_spectrum` in several tools, a single round trip would make
   the change permanent.
2. **It drops empty spectra.** `add_spectrum` catches its own `_EmptySpectrumError`, warns
   and returns `False` -- *before* consulting the `on_error="error"` setting it was given.
   `filter_peaks` can emit an empty spectrum today, so this would silently lose pixels and
   desynchronise the spectrum count from the coordinate list. Handled in `ImzmlWriter`,
   which refuses an empty array up front and treats a `False` return as an error.

# upstream: both are worth reporting; see ROADMAP.md, Phase D gaps (4) and (5).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from imzy import IMZMLWriter
from imzy._writers._imzml import MZML_NAMESPACE

if TYPE_CHECKING:
    from collections.abc import Sequence
    from xml.etree import ElementTree as ET

_POSITION_Z = "IMS:1000052"
_SCAN_PATH = f"{{{MZML_NAMESPACE}}}scanList/{{{MZML_NAMESPACE}}}scan"


class DepictionIMZMLWriter(IMZMLWriter):
    """An `imzy.IMZMLWriter` that keeps a 2D acquisition 2D."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._wrote_3d_coordinates = False

    def add_spectrum(
        self,
        mzs: Sequence[float],
        intensities: Sequence[float],
        coords: Sequence[int | float],
        *args: Any,
        **kwargs: Any,
    ) -> bool:
        written = super().add_spectrum(mzs, intensities, coords, *args, **kwargs)
        # Only a spectrum that was actually written may influence the output, so a rejected
        # 3D spectrum must not turn the file into a 3D one.
        if written and len(coords) == 3:
            self._wrote_3d_coordinates = True
        return written

    def _add_scan_list(self, spectrum_element: ET.Element, spectrum: Any) -> None:
        super()._add_scan_list(spectrum_element, spectrum)
        if self._wrote_3d_coordinates:
            return
        scan = spectrum_element.find(_SCAN_PATH)
        if scan is None:
            raise RuntimeError(f"imzy no longer writes {_SCAN_PATH}; the 2D coordinate patch needs revisiting")
        position_z = [child for child in scan if child.attrib.get("accession") == _POSITION_Z]
        if len(position_z) != 1:
            raise RuntimeError(
                f"Expected exactly one {_POSITION_Z} cvParam to remove, found {len(position_z)};"
                f" the 2D coordinate patch needs revisiting"
            )
        scan.remove(position_z[0])
