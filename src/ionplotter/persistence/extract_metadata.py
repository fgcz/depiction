from functools import cached_property
from pathlib import Path
from pydantic import BaseModel
import lxml.etree

from ionplotter.persistence.pixel_size import PixelSize


class Metadata(BaseModel):
    pixel_size: PixelSize
    data_processing: list[str]
    softwares: list[str]


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
            softwares=extractor.softwares(),
        )

    @cached_property
    def _etree(self) -> lxml.etree.ElementTree:
        return lxml.etree.parse(str(self._imzml_path))

    def pixel_size(self) -> PixelSize | None:
        collect = {
            "pixel_size_x": [
                float(el.attrib["value"])
                for el in self._etree.findall(f".//{self._ns}cvParam[@accession='IMS:1000046']")
            ],
            "pixel_size_y": [
                float(el.attrib["value"])
                for el in self._etree.findall(f".//{self._ns}cvParam[@accession='IMS:1000047']")
            ],
        }
        if len(collect["pixel_size_x"]) == 0:
            return None
        if len(collect["pixel_size_x"]) > 1 or len(collect["pixel_size_y"]) > 1:
            raise NotImplementedError("Multiple pixel sizes found: {collect}")

        # TODO actually check the units (they are missing in the files i have seen so far)
        unit = "micrometer"

        [pixel_size_x] = collect["pixel_size_x"]
        if len(collect["pixel_size_y"]) == 0:
            return PixelSize(size_x=pixel_size_x, size_y=pixel_size_x, unit=unit)
        [pixel_size_y] = collect["pixel_size_y"]
        return PixelSize(size_x=pixel_size_x, size_y=pixel_size_y, unit=unit)

    def data_processing(self) -> list[str]:
        # each method will have some child accessions, for now we just parse it all into a flat string
        # this should probably be improved in the future!! (but then we will simply upgrade to a different type that
        # can hold the extra information, so it should be obvious whether the new code is used)
        items = self._etree.findall(f".//{self._ns}processingMethod/{self._ns}cvParam")
        return [item.attrib["name"] for item in items]

    def softwares(self) -> list[str]:
        items = self._etree.findall(f".//{self._ns}software/{self._ns}cvParam")
        return [item.attrib["name"] for item in items]
