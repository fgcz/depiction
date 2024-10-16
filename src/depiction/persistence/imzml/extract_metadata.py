import warnings
from functools import cached_property
from pathlib import Path
from xml.etree.ElementTree import ElementTree

import lxml.etree
from pydantic import BaseModel

from depiction.persistence.imzml.parser.parse_metadata import ParseMetadata
from depiction.persistence.pixel_size import PixelSize


class Metadata(BaseModel):
    pixel_size: PixelSize
    data_processing: list[str]
    software: list[str]


class ExtractMetadata:
    """Handles the extraction of some metadata that is used for the data processing."""

    def __init__(self, imzml_path: Path) -> None:
        self._imzml_path = Path(imzml_path)
        self._ns = "{http://psi.hupo.org/ms/mzml}"

    @classmethod
    def extract_file(cls, path: Path) -> Metadata:
        extractor = cls(path)
        return Metadata(
            pixel_size=extractor.pixel_size(),
            data_processing=extractor.data_processing(),
            software=extractor.software(),
        )

    @cached_property
    def _etree(self) -> lxml.etree.ElementTree:
        return lxml.etree.parse(str(self._imzml_path))

    def pixel_size(self) -> PixelSize | None:
        # TODO delete
        warnings.warn("This method is deprecated, use `ParseMetadata.pixel_size` instead", DeprecationWarning)
        etree = ElementTree(file=self._imzml_path)
        return ParseMetadata(etree).pixel_size

    def data_processing(self) -> list[str]:
        # each method will have some child accessions, for now we just parse it all into a flat string
        # this should probably be improved in the future!! (but then we will simply upgrade to a different type that
        # can hold the extra information, so it should be obvious whether the new code is used)
        items = self._etree.findall(f".//{self._ns}processingMethod/{self._ns}cvParam")
        return [item.attrib["name"] for item in items]

    def software(self) -> list[str]:
        items = self._etree.findall(f".//{self._ns}software/{self._ns}cvParam")
        return [item.attrib["name"] for item in items]
