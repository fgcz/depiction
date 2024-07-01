from __future__ import annotations
import enum


class ImzmlModeEnum(enum.Enum):
    """Enum for the different modes of imzml files."""

    CONTINUOUS = enum.auto()
    PROCESSED = enum.auto()

    @classmethod
    def as_pyimzml_str(cls, instance: ImzmlModeEnum) -> str:
        """Returns the string representation of the enum value as used by pyimzml."""
        return instance.name.lower()

    @classmethod
    def from_pyimzml_str(cls, value: str) -> ImzmlModeEnum:
        if value == "continuous":
            return cls.CONTINUOUS
        elif value == "processed":
            return cls.PROCESSED
        else:
            raise ValueError(f'Unknown imzml mode "{value}".')
