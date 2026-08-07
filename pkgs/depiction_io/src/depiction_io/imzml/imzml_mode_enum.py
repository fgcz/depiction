from __future__ import annotations
import enum


class ImzmlModeEnum(enum.Enum):
    """Represents the different modes of imzml files."""

    CONTINUOUS = enum.auto()
    """Continuous mode imzML files, they share the same m/z values for all spectra."""

    PROCESSED = enum.auto()
    """Processed mode imzML files, they can have different m/z values for each spectrum."""

    @classmethod
    def as_imzml_str(cls, instance: ImzmlModeEnum) -> str:
        return instance.name.lower()

    @classmethod
    def from_imzml_str(cls, value: str) -> ImzmlModeEnum:
        if value == "continuous":
            return cls.CONTINUOUS
        elif value == "processed":
            return cls.PROCESSED
        else:
            raise ValueError(f'Unknown imzml mode "{value}".')
