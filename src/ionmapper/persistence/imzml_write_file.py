import os
from contextlib import contextmanager
from pathlib import Path

from ionmapper.persistence.imzml_mode_enum import ImzmlModeEnum
from ionmapper.persistence.imzml_writer import ImzmlWriter


class ImzmlWriteFile:
    """A handle for a .imzML file that is to be written.

    Args:
        path: The path to the .imzML file.
        imzml_mode: The mode of the .imzML file.
        write_mode: The write mode. If "x", the file must not exist. If "w", the file will be overwritten if it exists.
            Other values are not supported.
    """

    def __init__(self, path: str | Path, imzml_mode: ImzmlModeEnum, write_mode: str = "x"):
        self._path = str(path)
        self._imzml_mode = imzml_mode
        self._write_mode = write_mode

    @property
    def imzml_file(self) -> str:
        """The path to the underlying .imzML file."""
        if not self._path.lower().endswith(".imzml"):
            raise ValueError(f"Expected .imzML file, got {self._path}")
        return self._path

    @property
    def ibd_file(self) -> str:
        """The path to the accompanying .ibd file."""
        return self._path.replace(".imzML", ".ibd")

    @property
    def imzml_mode(self) -> ImzmlModeEnum:
        """The imzml mode of the .imzML file."""
        return self._imzml_mode

    @contextmanager
    def writer(self):
        """Opens the .imzML file for writing and yields an `ImzmlWriter` instance."""
        if self._write_mode == "x":
            if os.path.exists(self.imzml_file):
                raise ValueError(f"File {self.imzml_file} already exists.")
        elif self._write_mode == "w":
            if os.path.exists(self.imzml_file):
                # TODO make the handling more robust, analogous to the changes in ImzmlReadFile, however there might
                #      need to be a bigger refactoring in the future anyhow.
                os.remove(self.imzml_file)
                os.remove(self.ibd_file)
        else:
            raise ValueError(f"Invalid write mode: {self._write_mode!r}")

        writer = ImzmlWriter.open(path=self.imzml_file, imzml_mode=self._imzml_mode)
        try:
            yield writer
        finally:
            writer.close()

    def __repr__(self) -> str:
        return f"ImzmlWriteFile(path={self._path!r}, imzml_mode={self._imzml_mode!r}, write_mode={self._write_mode!r})"
