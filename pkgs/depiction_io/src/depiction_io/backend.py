"""Chooses which reader implementation opens a file.

The legacy `ImzmlReadFile` is the default and stays the default: the imzy backend is not
yet validated against real acquisitions (see `docs/refactoring/ROADMAP.md`, Phase E), so it
is opt-in per call or through the `DEPICTION_IO_BACKEND` environment variable.

Note that the tools in `depiction` still construct `ImzmlReadFile` directly rather than
going through this function, so the environment variable does not currently redirect them.
Routing those call sites through here is the first step of Phase E, at which point flipping
the default becomes a one-line change.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from depiction_io.imzml.imzml_read_file import ImzmlReadFile
from depiction_io.imzy_backend.imzy_read_file import ImzyReadFile

if TYPE_CHECKING:
    from depiction_io.types import GenericReadFile

ReadBackend = Literal["legacy", "imzy"]

#: Overrides the default backend for .imzML files.
BACKEND_ENV_VAR = "DEPICTION_IO_BACKEND"

DEFAULT_BACKEND: ReadBackend = "legacy"


def get_read_file(path: str | Path, backend: ReadBackend | None = None) -> GenericReadFile:
    """Returns a read file for `path`, using the configured backend.

    Args:
        path: the file to open. `.imzML` is served by either backend; anything else (a Bruker
            `.d` directory, say) can only be served by imzy.
        backend: forces a backend, overriding `DEPICTION_IO_BACKEND`.

    Raises:
        ValueError: the requested backend does not exist.
        RuntimeError: a vendor format was requested on a platform where imzy cannot read it.
    """
    path = Path(path)
    if backend is None:
        backend = os.environ.get(BACKEND_ENV_VAR, DEFAULT_BACKEND)
    if backend not in ("legacy", "imzy"):
        raise ValueError(f"Unknown backend {backend!r}, expected 'legacy' or 'imzy' (from ${BACKEND_ENV_VAR})")

    if path.suffix.lower() != ".imzml":
        # Nothing else can read a vendor format, so the requested backend is not consulted.
        if sys.platform == "darwin":
            raise RuntimeError(
                f"Cannot read {path}: imzy disables its Bruker readers on macOS (see imzy/plugins.py), and the"
                f" legacy backend reads .imzML only."
            )
        return ImzyReadFile(path)

    return ImzyReadFile(path) if backend == "imzy" else ImzmlReadFile(path)
