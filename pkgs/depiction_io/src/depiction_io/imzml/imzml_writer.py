from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from depiction_io.imzml.imzml_alignment_tracker import ImzmlAlignmentTracker
from depiction_io.imzml.imzml_mode_enum import ImzmlModeEnum
from depiction_io.imzy_backend.imzy_writer import DepictionIMZMLWriter
from depiction_io.types import GenericWriter

if TYPE_CHECKING:
    from numpy.typing import NDArray


class ImzmlWriter(GenericWriter):
    """Writes .imzML files through `imzy`.

    `ImzmlAlignmentTracker` is kept even though imzy performs its own continuous-mode check,
    because imzy's check happens on the second spectrum and phrases the failure differently;
    keeping this one means the error a caller sees does not change.
    """

    def __init__(
        self,
        *,
        wrapped_imzml_writer: DepictionIMZMLWriter,
        imzml_alignment_tracker: ImzmlAlignmentTracker | None,
    ) -> None:
        self._imzml_writer = wrapped_imzml_writer
        self._imzml_alignment_tracker = imzml_alignment_tracker

    @classmethod
    def open(
        cls,
        path: str | Path,
        imzml_mode: ImzmlModeEnum,
        imzml_alignment_tracking: bool = True,
        mz_dtype: np.typing.DTypeLike = np.float64,
        intensity_dtype: np.typing.DTypeLike = np.float32,
        overwrite: bool = False,
    ) -> ImzmlWriter:
        """Opens an imzML file."""
        imzml_alignment_tracker = ImzmlAlignmentTracker() if imzml_alignment_tracking else None
        # imzy rewrites the suffix to `.imzML` rather than using the path it was given, so on
        # a case-sensitive filesystem a caller who asked for `out.imzml` would find nothing
        # there and the data in `out.imzML`. pyimzml wrote the name verbatim.
        if Path(path).suffix != ".imzML":
            raise ValueError(f"Expected a path ending in '.imzML', got {path!r}; imzy would write elsewhere.")
        return cls(
            wrapped_imzml_writer=DepictionIMZMLWriter(
                str(path),
                mz_dtype=mz_dtype,
                intensity_dtype=intensity_dtype,
                ibd_mode=ImzmlModeEnum.as_imzml_str(imzml_mode),
                # depiction's coordinates are 1-based already, so imzy must not shift them.
                coordinate_origin="one",
                # Made explicit because the empty-spectrum guard below exists precisely
                # because imzy does not honour this setting for empty spectra.
                on_error="error",
                overwrite=overwrite,
            ),
            imzml_alignment_tracker=imzml_alignment_tracker,
        )

    def close(self) -> None:
        """Writes the imzML and closes the file.

        NOTE: closing a writer that received no spectrum raises. pyimzml used to write a
        malformed file instead, which was worse, but a caller that opens a writer inside a
        `with` block and then finds nothing to write now has to handle it.
        """
        self._imzml_writer.close()

    def discard(self) -> None:
        """Closes the writer, throwing away everything written so far.

        The way out for a caller that cannot finish: `close()` renames whatever was written so
        far into place, which for an abandoned write means a truncated file sitting where a
        complete one is expected.
        """
        self._imzml_writer.discard()

    def deactivate_alignment_tracker(self) -> None:
        self._imzml_alignment_tracker = None

    @property
    def imzml_mode(self) -> ImzmlModeEnum:
        """Returns the mode of the imzML file."""
        return ImzmlModeEnum.from_imzml_str(self._imzml_writer.ibd_mode)

    @property
    def imzml_path(self) -> Path:
        return Path(self._imzml_writer.imzml_path)

    @property
    def ibd_path(self) -> Path:
        return Path(self._imzml_writer.ibd_path)

    @property
    def is_aligned(self) -> bool:
        """Returns True if the spectra are aligned."""
        return self._imzml_alignment_tracker.is_aligned

    def add_spectrum(
        self,
        mz_arr: NDArray[np.float64],
        int_arr: NDArray[np.float64],
        coordinates: tuple[int, int] | tuple[int, int, int] | NDArray[np.int64],
    ) -> None:
        if len(mz_arr) != len(int_arr):
            raise ValueError(f"{len(mz_arr)=} and {len(int_arr)=} must be equal.")
        if len(mz_arr) == 0:
            # imzy would warn and skip, leaving n_spectra out of step with the coordinates
            # the caller believes it wrote. See `imzy_writer` for the full story.
            raise ValueError(f"Refusing to write an empty spectrum at {tuple(coordinates)}.")

        # Handle alignment check information.
        if self._imzml_alignment_tracker:
            self._imzml_alignment_tracker.track_mz_array(mz_arr)

        if self.imzml_mode == ImzmlModeEnum.CONTINUOUS and self._imzml_alignment_tracker and not self.is_aligned:
            raise ValueError(
                "The m/z array of the first spectrum must be identical to the m/z array of all other spectra!"
            )

        # Write the spectrum.
        if not self._imzml_writer.add_spectrum(mz_arr, int_arr, coordinates):
            raise RuntimeError(f"imzy declined to write the spectrum at {tuple(coordinates)}.")
