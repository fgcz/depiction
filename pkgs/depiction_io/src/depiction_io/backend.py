"""Chooses which reader implementation opens a file.

There is only one now -- `imzy`, behind `ImzyReadFile`. This function stays because the
choice it used to make is still a real one for anything that is not an .imzML: imzy's
Bruker readers are disabled on macOS, and that is worth saying plainly rather than failing
somewhere further in.

Keep constructing read files through here rather than naming `ImzyReadFile` directly. It is
the seam that made swapping the imzML parser out a one-line change, and it is what a
successor adding a format would reach for.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from depiction_io.imzy_backend.imzy_read_file import ImzyReadFile

if TYPE_CHECKING:
    from depiction_io.types import GenericReadFile


def get_read_file(path: str | Path) -> GenericReadFile:
    """Returns a read file for `path`.

    Args:
        path: the file to open -- an `.imzML`, or a vendor format such as a Bruker `.d`
            directory.

    Raises:
        RuntimeError: a vendor format was requested on a platform where imzy cannot read it.
    """
    path = Path(path)
    if path.suffix.lower() != ".imzml" and sys.platform == "darwin":
        raise RuntimeError(f"Cannot read {path}: imzy disables its Bruker readers on macOS (see imzy/plugins.py).")
    return ImzyReadFile(path)
