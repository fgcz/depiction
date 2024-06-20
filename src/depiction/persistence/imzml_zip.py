from functools import cached_property
from pathlib import Path


class ImzmlZip:
    """Handles imzml data stored in a zip alongside its ibd file.
    A file must contain exactly one .imzML file and exactly one .ibd file,
    sharing the same path except for the extension.
    They can however be in the root or in a subdirectory, allowing for greater compatibility.
    TODO when implementing a writer define a standard output format
    """

    def __init__(self, path: Path) -> None:
        self.path = path

    @cached_property
    def imzml_filename(self) -> str | None:
        pass
