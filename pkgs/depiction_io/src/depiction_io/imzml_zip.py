from functools import cached_property
from pathlib import Path
from zipfile import ZipFile

from loguru import logger


class ImzmlZip:
    """Handles imzml data stored in a zip alongside its ibd file.
    A file must contain exactly one .imzML file and exactly one .ibd file,
    sharing the same path except for the extension.
    They can however be in the root or in a subdirectory, allowing for greater compatibility.
    TODO when implementing a writer define a standard output format
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def imzml_filename(self) -> str | None:
        return self._entry_name[0] if self._entry_name is not None else None

    @property
    def ibd_filename(self) -> str | None:
        return self._entry_name[1] if self._entry_name is not None else None

    @cached_property
    def _entry_name(self) -> tuple[str, str] | None:
        with ZipFile(self.path, "r") as file:
            imzml_files = [name for name in file.namelist() if name.endswith(".imzML")]
            if len(imzml_files) != 1:
                logger.error(f"Expected exactly one .imzML file in {self.path}. Actual: {len(imzml_files)}")
                return None
            imzml_file = Path(imzml_files[0])
            ibd_file = imzml_file.with_suffix(".ibd")
            if str(ibd_file) not in file.namelist():
                logger.error(f"Expected {ibd_file} to be in the same zip file as {imzml_file}")
                return None
            return str(imzml_file), str(ibd_file)

    def extract(self, directory: Path | str, imzml_filename: Path | None = None) -> Path:
        """Extracts the .imzML and .ibd file to the given directory and returns the path to the .imzML file.
        If a filename is passed it will be used instead of the one determined by imzml_filename.
        """
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        with ZipFile(self.path, "r") as file:
            if imzml_filename is None:
                imzml_filename = directory / Path(self.imzml_filename).name
            file.extract(self.imzml_filename, directory / imzml_filename)
            file.extract(self.ibd_filename, directory / imzml_filename.with_suffix(".ibd"))
        return imzml_filename

    def __repr__(self) -> str:
        return f"ImzmlZip({self.path})"
