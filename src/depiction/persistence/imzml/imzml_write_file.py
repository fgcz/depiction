from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING
from collections.abc import Generator

from depiction.persistence.imzml.imzml_writer import ImzmlWriter
from depiction.persistence.types import GenericWriteFile

if TYPE_CHECKING:
    from depiction.persistence.imzml.imzml_mode_enum import ImzmlModeEnum


class ImzmlWriteFile(GenericWriteFile):
    """A handle for a .imzML file that is to be written.

    Args:
        path: The path to the .imzML file.
        imzml_mode: The mode of the .imzML file.
        write_mode: The write mode. If "x", the file must not exist. If "w", the file will be overwritten if it exists.
            Other values are not supported.
    """

    def __init__(self, path: str | Path, imzml_mode: ImzmlModeEnum, write_mode: str = "x") -> None:
        self._path = Path(path)
        self._imzml_mode = imzml_mode
        self._write_mode = write_mode

    @property
    def imzml_file(self) -> Path:
        """The path to the underlying .imzML file."""
        if self._path.suffix.lower() != ".imzml":
            raise ValueError(f"Expected .imzML file, got {self._path}")
        return self._path

    @property
    def ibd_file(self) -> Path:
        """The path to the accompanying .ibd file."""
        return self._path.with_suffix(".ibd")

    @property
    def imzml_mode(self) -> ImzmlModeEnum:
        """The imzml mode of the .imzML file."""
        return self._imzml_mode

    @contextmanager
    def writer(self) -> Generator[ImzmlWriter, None, None]:
        """Opens the .imzML file for writing and yields an `ImzmlWriter` instance."""
        if self._write_mode == "x":
            if self.imzml_file.exists():
                raise ValueError(f"File {self.imzml_file} already exists.")
        elif self._write_mode == "w":
            if self.imzml_file.exists():
                # TODO make the handling more robust, analogous to the changes in ImzmlReadFile, however there might
                #      need to be a bigger refactoring in the future anyhow.
                self.imzml_file.unlink()
                self.ibd_file.unlink()
        else:
            raise ValueError(f"Invalid write mode: {self._write_mode!r}")

        writer = ImzmlWriter.open(path=self.imzml_file, imzml_mode=self._imzml_mode)
        try:
            yield writer
        finally:
            writer.close()

    def __repr__(self) -> str:
        return f"ImzmlWriteFile(path={self._path!r}, imzml_mode={self._imzml_mode!r}, write_mode={self._write_mode!r})"
