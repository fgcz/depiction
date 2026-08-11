import shutil
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

    def extract(self, directory: Path | str, imzml_filename: Path | str | None = None) -> Path:
        """Extracts the .imzML and .ibd file into the given directory and returns the path to the .imzML file.

        If a filename is passed it replaces the one taken from the archive; it is resolved
        relative to `directory`. The two files are always written side by side, whatever
        layout the archive uses internally.
        """
        if self._entry_name is None:
            raise ValueError(f"Expected exactly one .imzML/.ibd pair in {self.path}")
        directory = Path(directory)
        name = Path(self.imzml_filename).name if imzml_filename is None else imzml_filename
        imzml_path = directory / name
        ibd_path = imzml_path.with_suffix(".ibd")
        imzml_path.parent.mkdir(parents=True, exist_ok=True)
        # Not `ZipFile.extract`, whose second argument is the directory to extract *into*, not
        # the destination filename -- it would create a directory named `<something>.imzML`
        # and write the member inside it under its archive-internal path.
        with ZipFile(self.path, "r") as file:
            for member, destination in ((self.imzml_filename, imzml_path), (self.ibd_filename, ibd_path)):
                with file.open(member) as source, destination.open("wb") as target:
                    shutil.copyfileobj(source, target)
        return imzml_path

    def __repr__(self) -> str:
        return f"ImzmlZip({self.path})"
