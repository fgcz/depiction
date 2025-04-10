from __future__ import annotations

import enum


class ImzmlModeEnum(enum.Enum):
    """Represents the different modes of imzml files."""

    CONTINUOUS = "continuous"
    """Continuous mode imzML files, they share the same m/z values for all spectra."""

    PROCESSED = "processed"
    """Processed mode imzML files, they can have different m/z values for each spectrum."""

    @classmethod
    def as_pyimzml_str(cls, instance: ImzmlModeEnum) -> str:
        return instance.name.lower()

    @classmethod
    def from_pyimzml_str(cls, value: str) -> ImzmlModeEnum:
        if value == "continuous":
            return cls.CONTINUOUS
        elif value == "processed":
            return cls.PROCESSED
        else:
            raise ValueError(f'Unknown imzml mode "{value}".')
