from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from depiction_io.imzml.imzml_writer import ImzmlWriter
from depiction_io.types import GenericWriteFile

if TYPE_CHECKING:
    from depiction_io.imzml.imzml_mode_enum import ImzmlModeEnum


class ImzmlWriteFile(GenericWriteFile):
    """A handle for a .imzML file that is to be written.

    Args:
        path: The path to the .imzML file.
        imzml_mode: The mode of the .imzML file.
        write_mode: The write mode. If "x", the file must not exist. If "w", the file will be overwritten if it exists.
            Other values are not supported.
    """

    def __init__(
        self,
        path: str | Path,
        imzml_mode: ImzmlModeEnum,
        write_mode: str = "x",
        mz_dtype: np.typing.DTypeLike = np.float64,
        intensity_dtype: np.typing.DTypeLike = np.float32,
    ) -> None:
        self._path = Path(path)
        self._imzml_mode = imzml_mode
        self._write_mode = write_mode
        self._mz_dtype = mz_dtype
        self._intensity_dtype = intensity_dtype

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
    def icache_file(self) -> Path:
        """The path of the offset cache imzy would write beside this file."""
        return self._path.with_suffix(".icache")

    @property
    def imzml_mode(self) -> ImzmlModeEnum:
        """The imzml mode of the .imzML file."""
        return self._imzml_mode

    @contextmanager
    def writer(self) -> Generator[ImzmlWriter]:
        """Opens the .imzML file for writing and yields an `ImzmlWriter` instance."""
        if self._write_mode == "x":
            if self.imzml_file.exists():
                raise ValueError(f"File {self.imzml_file} already exists.")
        elif self._write_mode == "w":
            if self.imzml_file.exists():
                # TODO this unlinks before checking that the writer can be opened, so a failure
                #      here leaves neither the old file nor a new one.
                self.imzml_file.unlink()
                self.ibd_file.unlink()
        else:
            raise ValueError(f"Invalid write mode: {self._write_mode!r}")

        # imzy caches an offset table in `<stem>.icache` beside whatever it reads, and
        # reloads it on sight without checking that it still matches the file. Leaving a
        # stale one behind means the next imzy read of this path reports the *previous*
        # file's spectrum count and coordinates while slicing the *new* .ibd -- silently.
        # See `imzy_reader` for the read-side half of this guard.
        self.icache_file.unlink(missing_ok=True)

        writer = ImzmlWriter.open(
            path=self.imzml_file,
            imzml_mode=self._imzml_mode,
            mz_dtype=self._mz_dtype,
            intensity_dtype=self._intensity_dtype,
            # The checks above already cleared the way; this only stops imzy from refusing
            # to start over a leftover .ibd whose .imzML was removed.
            overwrite=self._write_mode == "w",
        )
        try:
            yield writer
        except BaseException:
            # Closing an empty writer raises, which would replace whatever went wrong in
            # the body with a misleading "no spectra" error.
            with suppress(Exception):
                writer.close()
            raise
        else:
            writer.close()

    def __repr__(self) -> str:
        return f"ImzmlWriteFile(path={self._path!r}, imzml_mode={self._imzml_mode!r}, write_mode={self._write_mode!r})"
