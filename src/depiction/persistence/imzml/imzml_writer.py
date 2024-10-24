from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pyimzml
import pyimzml.ImzMLWriter

from depiction.persistence.imzml.imzml_alignment_tracker import ImzmlAlignmentTracker
from depiction.persistence.imzml.imzml_mode_enum import ImzmlModeEnum
from depiction.persistence.types import GenericWriter

if TYPE_CHECKING:
    import numpy as np


class ImzmlWriter(GenericWriter):
    def __init__(
        self,
        *,
        wrapped_imzml_writer: pyimzml.ImzMLWriter.ImzMLWriter,
        imzml_alignment_tracker: ImzmlAlignmentTracker | None,
    ) -> None:
        self._imzml_writer = wrapped_imzml_writer
        self._imzml_alignment_tracker = imzml_alignment_tracker

    @classmethod
    def open(cls, path: str | Path, imzml_mode: ImzmlModeEnum, imzml_alignment_tracking: bool = True) -> ImzmlWriter:
        """Opens an imzML file."""
        imzml_alignment_tracker = ImzmlAlignmentTracker() if imzml_alignment_tracking else None
        return cls(
            wrapped_imzml_writer=pyimzml.ImzMLWriter.ImzMLWriter(
                str(path),
                mode=ImzmlModeEnum.as_pyimzml_str(imzml_mode),
            ),
            imzml_alignment_tracker=imzml_alignment_tracker,
        )

    # TODO this currently has a bug, when closing a reader to which not a single spectrum was added.
    #      it should be handled gracefully, since there are reasonable cases,
    #      where this could occur with the use of "with" clauses.
    def close(self) -> None:
        self._imzml_writer.close()

    def deactivate_alignment_tracker(self) -> None:
        self._imzml_alignment_tracker = None

    @property
    def imzml_mode(self) -> ImzmlModeEnum:
        """Returns the mode of the imzML file."""
        return ImzmlModeEnum.from_pyimzml_str(self._imzml_writer.mode)

    @property
    def imzml_path(self) -> Path:
        return Path(self._imzml_writer.filename)

    @property
    def ibd_path(self) -> Path:
        return Path(self._imzml_writer.ibd_filename)

    @property
    def is_aligned(self) -> bool:
        """Returns True if the spectra are aligned."""
        return self._imzml_alignment_tracker.is_aligned

    def add_spectrum(
        self,
        mz_arr: np.ndarray,
        int_arr: np.ndarray,
        coordinates: tuple[int, int] | tuple[int, int, int],
    ) -> None:
        if len(mz_arr) != len(int_arr):
            raise ValueError(f"{len(mz_arr)=} and {len(int_arr)=} must be equal.")

        # Handle alignment check information.
        if self._imzml_alignment_tracker:
            self._imzml_alignment_tracker.track_mz_array(mz_arr)

        if self.imzml_mode == ImzmlModeEnum.CONTINUOUS and self._imzml_alignment_tracker and not self.is_aligned:
            raise ValueError(
                "The m/z array of the first spectrum must be identical to the m/z array of all other spectra!"
            )

        # Write the spectrum.
        self._imzml_writer.addSpectrum(mz_arr, int_arr, coordinates)
