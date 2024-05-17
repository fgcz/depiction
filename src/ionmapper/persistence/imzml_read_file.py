import os
from collections import defaultdict
from contextlib import contextmanager
from functools import cached_property
from pathlib import Path
from typing import Any, Optional
from collections.abc import Generator

import pyimzml.ImzMLParser
from ionmapper.persistence.file_checksums import FileChecksums
from ionmapper.persistence.imzml_mode_enum import ImzmlModeEnum
from ionmapper.persistence.imzml_reader import ImzmlReader
from numpy.typing import NDArray

from ionmapper.persistence.pixel_size import PixelSize


class ImzmlReadFile:
    """Represents a .imzML file and its accompanying .ibd file.
    It provides several methods and properties to obtain general information about this file and verify its integrity.
    However, to load the actual spectra, use the `reader` context manager or `get_reader` method to obtain a
    `ImzmlReader` instance which provides the relevant functionality.
    """

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)

    @property
    def imzml_file(self) -> str:
        """Returns the path to the underlying .imzML file."""
        if not self._path.lower().endswith(".imzml"):
            raise ValueError(f"Expected .imzML file, got {self._path}")
        return self._path

    @property
    def ibd_file(self) -> str:
        """Returns the path to the accompanying .ibd file."""
        if not self._path.lower().endswith(".imzml"):
            raise ValueError(f"Expected .imzML file, got {self._path}")
        return self._path[:-6] + ".ibd"

    @contextmanager
    def reader(self) -> Generator[ImzmlReader, None, None]:
        """Returns a context manager that yields an `ImzmlReader` instance."""
        reader = self.get_reader()
        try:
            yield reader
        finally:
            reader.close()

    def get_reader(self) -> ImzmlReader:
        portable_reader = self._get_pyimzml_portable_reader
        return ImzmlReader(portable_reader=portable_reader, imzml_path=self._path)

    @cached_property
    def _get_pyimzml_portable_reader(
        self,
    ) -> pyimzml.ImzMLParser.PortableSpectrumReader:
        with pyimzml.ImzMLParser.ImzMLParser(self._path) as parser:
            return parser.portable_spectrum_reader()

    @cached_property
    def n_spectra(self) -> int:
        """Returns the number of spectra in the .imzML file."""
        return self._cached_properties["n_spectra"]

    @cached_property
    def imzml_mode(self) -> ImzmlModeEnum:
        """Returns the mode of the .imzML file (continuous or processed)."""
        return self._cached_properties["imzml_mode"]

    @cached_property
    def coordinates(self) -> NDArray[int]:
        """Returns the spatial coordinates of the spectra in the .imzML file.
        Shape: (n_spectra, n_dimensions) where n_dimensions is 2 or 3 depending on the file."""
        # TODO check if it isn't simply always 3d because of pyimzml
        return self._cached_properties["coordinates"]

    @property
    def coordinates_2d(self) -> NDArray[int]:
        """Returns the spatial coordinates of the spectra in the .imzML file.
        Shape: (n_spectra, 2) where the first two columns are the x and y coordinates."""
        # TODO double check convention and update docstring accordingly
        return self.coordinates[:, :2]

    @property
    def compact_metadata(self) -> dict[str, int | str | list[float]]:
        """Returns a compact representation of general metadata about the .imzML file, useful when comparing a large
        number of files."""
        return {
            "n_spectra": self.n_spectra,
            "imzml_mode": self.imzml_mode.name,
            "coordinate_extent": list(self.coordinates.max(0) - self.coordinates.min(0) + 1),
            "imzml_file": self.imzml_file,
            "ibd_file": self.ibd_file,
        }

    @cached_property
    def metadata_checksums(self) -> dict[str, str]:
        # from the mzML standard:
        #   e.g.: MS:1000568 (MD5)
        #   e.g.: MS:1000569 (SHA-1)
        checksums = {}
        with pyimzml.ImzMLParser.ImzMLParser(self._path) as parser:
            if "ibd MD5" in parser.metadata.file_description:
                checksums["md5"] = parser.metadata.file_description["ibd MD5"].lower()
            if "ibd SHA-1" in parser.metadata.file_description:
                checksums["sha1"] = parser.metadata.file_description["ibd SHA-1"].lower()
            if "ibd SHA-256" in parser.metadata.file_description:
                checksums["sha256"] = parser.metadata.file_description["ibd SHA-256"].lower()
        return checksums

    @cached_property
    def ibd_checksums(self) -> FileChecksums:
        return FileChecksums(file_path=self.ibd_file)

    @cached_property
    def is_checksum_valid(self) -> Optional[bool]:
        """Returns True if the checksum of the .ibd file matches the expected value. False otherwise.
        This operation can be slow for large files, but will be cached after the first call.
        `None` is returned when checksum information is available.
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

    @cached_property
    def file_sizes_bytes(self) -> dict[str, int]:
        """Returns the sizes of the .imzML and .ibd files in bytes."""
        return {
            "imzml": os.path.getsize(self.imzml_file),
            "ibd": os.path.getsize(self.ibd_file),
        }

    @cached_property
    def file_sizes_mb(self) -> dict[str, float]:
        """Returns the sizes of the .imzML and .ibd files in MB."""
        file_sizes = self.file_sizes_bytes
        return {k: v / 1024**2 for k, v in file_sizes.items()}

    def summary(self, checksums: bool = True) -> str:
        if checksums:
            checksum_valid = self.is_checksum_valid
            if checksum_valid is None:
                checksum_valid = "Could not be determined (missing metadata)"
            checksum_line = f"is_checksum_valid: {checksum_valid}\n"
        else:
            checksum_line = ""

        if self.imzml_mode == ImzmlModeEnum.CONTINUOUS:
            with self.reader() as reader:
                mz_arr = reader.get_spectrum_mz(0)
            n_mz_bins = len(mz_arr)
            # TODO use reader.get_spectra_mz_range()
            mz_range_line = f"m/z range: {mz_arr.min():.2f} - {mz_arr.max():.2f} ({n_mz_bins} bins)\n"
        else:
            mz_range_line = ""

        file_sizes = self.file_sizes_mb
        return (
            f"imzML file: {self._path} ({file_sizes['imzml']:.2f} MB)\n"
            f"ibd file: {self.ibd_file} ({file_sizes['ibd']:.2f} MB)\n"
            f"imzML mode: {self.imzml_mode.name}\n"
            f"n_spectra: {self.n_spectra}\n"
            f"{checksum_line}"
            f"{mz_range_line}"
        )

    def print_summary(self, checksums: bool = True) -> None:
        print(self.summary(checksums=checksums))

    @cached_property
    def pixel_size(self) -> PixelSize | None:
        # TODO optimize and improve, error handling when missing
        # TODO also make available in the parser class
        with pyimzml.ImzMLParser.ImzMLParser(self._path) as parser:
            collect = defaultdict(set)
            for scan_settings in parser.metadata.pretty()["scan_settings"].values():
                if "pixel size (x)" not in scan_settings:
                    continue
                collect["pixel_size_x"].add(scan_settings["pixel size (x)"])
                collect["pixel_size_y"].add(scan_settings.get("pixel size y", scan_settings["pixel size (x)"]))
            if len(collect["pixel_size_x"]) == 0:
                return None
            [pixel_size_x] = collect["pixel_size_x"]
            [pixel_size_y] = collect["pixel_size_y"]
        # TODO actual check of the unit
        return PixelSize(size_x=pixel_size_x, size_y=pixel_size_y, unit="micrometer")

    @cached_property
    def _cached_properties(self) -> dict[str, Any]:
        with self.reader() as reader:
            return {
                "n_spectra": reader.n_spectra,
                "imzml_mode": reader.imzml_mode,
                "coordinates": reader.coordinates,
            }

    def __repr__(self) -> str:
        return f"ImzmlReadFile({self._path!r})"
