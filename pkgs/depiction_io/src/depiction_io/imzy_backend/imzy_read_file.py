from __future__ import annotations

import shutil
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any
from xml.etree.ElementTree import ElementTree

import numpy as np

from depiction_io.file_checksums import FileChecksums
from depiction_io.imzml.parser.parse_metadata import ParseMetadata
from depiction_io.imzy_backend.imzml_scan import ImzmlScan, scan_imzml
from depiction_io.imzy_backend.imzy_reader import ImzyReader
from depiction_io.types import GenericReadFile

if TYPE_CHECKING:
    from collections.abc import Generator

    from numpy.typing import NDArray

    from depiction_io.imzml.imzml_mode_enum import ImzmlModeEnum
    from depiction_io.pixel_size import PixelSize


class ImzyReadFile(GenericReadFile):
    """Represents a file readable by `imzy`, behind the `GenericReadFile` protocol.

    Construction is cheap and does no I/O, so instances can be handed to worker processes.
    The imzML scan that guards against compressed input runs on the first access that needs
    a reader, and its result is kept on the instance so that it travels with the pickle
    rather than being repeated in every worker.

    Checksums and pixel size are read with `ParseMetadata`, not with imzy: imzy parses no
    checksums at all, and reports a pixel size of 1 where the file declares none, whereas
    `ImzmlReadFile` reports `None`. Both classes are already backend-independent, so reusing
    them is both less code and exact parity.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    @property
    def is_imzml(self) -> bool:
        """Whether this is an .imzML file, as opposed to a vendor format imzy can open."""
        return self._path.suffix.lower() == ".imzml"

    @property
    def imzml_file(self) -> Path:
        """Returns the path to the underlying .imzML file."""
        if not self.is_imzml:
            raise ValueError(f"Expected .imzML file, got {self._path}")
        return self._path

    @property
    def ibd_file(self) -> Path:
        """Returns the path to the accompanying .ibd file."""
        return self.imzml_file.with_suffix(".ibd")

    @cached_property
    def scan(self) -> ImzmlScan | None:
        """The result of the guard scan, or None for a non-imzML input."""
        return scan_imzml(self._path) if self.is_imzml else None

    @contextmanager
    def reader(self) -> Generator[ImzyReader]:
        """Returns a context manager that yields an `ImzyReader` instance."""
        reader = self.get_reader()
        try:
            yield reader
        finally:
            reader.close()

    def get_reader(self) -> ImzyReader:
        if self.scan is not None:
            self.scan.raise_if_unsupported_compression(self._path)
            return ImzyReader(self._path, declares_z=self.scan.declares_z)
        # A vendor format: there is no imzML to scan, and z always exists.
        return ImzyReader(self._path, declares_z=True)

    @cached_property
    def n_spectra(self) -> int:
        """Returns the number of spectra in the file."""
        return self._cached_properties["n_spectra"]

    @cached_property
    def imzml_mode(self) -> ImzmlModeEnum:
        """Returns the mode of the file (continuous or processed)."""
        return self._cached_properties["imzml_mode"]

    @cached_property
    def coordinates(self) -> NDArray[np.int64]:
        """Returns the spatial coordinates of the spectra.
        Shape: (n_spectra, n_dimensions) where n_dimensions is 2 or 3 depending on the file."""
        return self._cached_properties["coordinates"]

    @property
    def compact_metadata(self) -> dict[str, int | str | list[float]]:
        """Returns a compact representation of general metadata about the file, useful when comparing a large
        number of files."""
        return {
            "n_spectra": self.n_spectra,
            "imzml_mode": self.imzml_mode.name,
            "coordinate_extent": list(self.coordinates.max(0) - self.coordinates.min(0) + 1),
            "imzml_file": str(self._path),
            "ibd_file": str(self.ibd_file) if self.is_imzml else "",
        }

    @cached_property
    def metadata_checksums(self) -> dict[str, str]:
        if not self.is_imzml:
            return {}
        return ParseMetadata(etree=ElementTree(file=self._path)).ibd_checksums

    @cached_property
    def ibd_checksums(self) -> FileChecksums:
        return FileChecksums(file_path=self.ibd_file)

    @cached_property
    def is_checksum_valid(self) -> bool | None:
        """Returns True if the checksum of the .ibd file matches the expected value, False if it does not, and
        None when the file declares no checksum. This can be slow for large files, but is cached.
        """
        if not self.metadata_checksums:
            return None
        elif "sha1" in self.metadata_checksums:
            return self.metadata_checksums["sha1"] == self.ibd_checksums.checksum_sha1
        elif "sha256" in self.metadata_checksums:
            return self.metadata_checksums["sha256"] == self.ibd_checksums.checksum_sha256
        elif "md5" in self.metadata_checksums:
            return self.metadata_checksums["md5"] == self.ibd_checksums.checksum_md5
        else:
            raise ValueError(f"Invalid metadata_checksums: {self.metadata_checksums}")

    def summary(self, checksums: bool = True) -> str:
        if checksums:
            checksum_valid = self.is_checksum_valid
            if checksum_valid is None:
                checksum_valid = "Could not be determined (missing metadata)"
            checksum_line = f"is_checksum_valid: {checksum_valid}\n"
        else:
            checksum_line = ""
        return (
            f"file: {self._path}\n"
            f"imzML mode: {self.imzml_mode.name}\n"
            f"n_spectra: {self.n_spectra}\n"
            f"{checksum_line}"
        )

    @cached_property
    def pixel_size(self) -> PixelSize | None:
        """Returns the pixel size of the spectra, if the file declares one."""
        if not self.is_imzml:
            return None
        return ParseMetadata(etree=ElementTree(file=self._path)).pixel_size

    def copy_to(self, path: Path) -> None:
        """Copies the file of this instance to the given path. Needs to end with .imzML."""
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(self.imzml_file, path)
        shutil.copy(self.ibd_file, path.with_suffix(".ibd"))

    @cached_property
    def _cached_properties(self) -> dict[str, Any]:
        with self.reader() as reader:
            return {
                "n_spectra": reader.n_spectra,
                "imzml_mode": reader.imzml_mode,
                "coordinates": reader.coordinates,
            }

    def __repr__(self) -> str:
        return f"ImzyReadFile({str(self._path)!r})"
