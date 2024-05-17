from __future__ import annotations

import glob
from typing import TextIO

import pandas as pd


class FlexImagingInfoFile:
    """Parses a FlexImaging Info File and provides access to its attributes.
    This file is usually found in the same directory as the .d directory and named `*_info.txt`.
    """

    def __init__(self, attributes: dict[str, str]):
        self._attributes = attributes

    def keys(self) -> set[str]:
        return set(self._attributes.keys())

    def as_dict(self) -> dict[str, str]:
        return self._attributes.copy()

    def as_series(self) -> pd.Series:
        return pd.Series(self._attributes)

    @classmethod
    def parse_file(cls, file: TextIO) -> FlexImagingInfoFile:
        lines = file.readlines()
        # TODO when testing: check trailing space allowed
        if lines[0].strip() != "FlexImaging Info File":
            raise ValueError("Unsupported file format")
        attributes = {}
        for line in lines[1:]:
            key_raw, value = line.split(":", maxsplit=1)
            key_norm = cls.normalize_key(key_raw)
            if key_norm in attributes:
                raise RuntimeError(f"Duplicate key: {key_norm}")
            attributes[key_norm] = value.strip()
        return cls(attributes=attributes)

    @classmethod
    def parse_in_directory(cls, directory: str) -> FlexImagingInfoFile:
        [match] = glob.glob(f"{directory}/*_info.txt")
        with open(match, "r") as file:
            return cls.parse_file(file=file)

    @classmethod
    def normalize_key(cls, key: str) -> str:
        return key.strip().lower().replace(" ", "_")
