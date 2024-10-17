from __future__ import annotations

from pathlib import Path
from xml.etree.ElementTree import ElementTree

from depiction.persistence.imzml.metadata import Metadata
from depiction.persistence.pixel_size import PixelSize


class ParseMetadata:
    def __init__(self, etree: ElementTree) -> None:
        self._etree = etree
        self._ns = "{http://psi.hupo.org/ms/mzml}"

    @classmethod
    def from_file(cls, path: Path) -> ParseMetadata:
        return cls(ElementTree(file=path))

    def parse(self) -> Metadata:
        return Metadata(
            pixel_size=self.pixel_size,
            data_processing=self.data_processing,
            software=self.software,
            ibd_checksums=self.ibd_checksums,
        )

    @property
    def ibd_checksums(self) -> dict[str, str]:
        elements = self._etree.findall(f".//{self._ns}fileDescription/{self._ns}fileContent/{self._ns}cvParam")
        checksums = {}
        for element in elements:
            if element.attrib["accession"] in ("MS:1000568", "IMS:1000090"):
                checksums["md5"] = element.attrib["value"].lower()
            elif element.attrib["accession"] in ("MS:1000569", "IMS:1000091"):
                checksums["sha1"] = element.attrib["value"].lower()
            elif element.attrib["accession"] in ("MS:1003151", "IMS:1000092"):
                checksums["sha256"] = element.attrib["value"].lower()
        return checksums

    @property
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
            raise NotImplementedError(f"Multiple pixel sizes found: {collect}")

        # TODO actually check the units (they are missing in the files i have seen so far)
        unit = "micrometer"

        [pixel_size_x] = collect["pixel_size_x"]
        if not collect["pixel_size_y"]:
            return PixelSize(size_x=pixel_size_x, size_y=pixel_size_x, unit=unit)
        [pixel_size_y] = collect["pixel_size_y"]
        return PixelSize(size_x=pixel_size_x, size_y=pixel_size_y, unit=unit)

    @property
    def data_processing(self) -> list[str]:
        # each method will have some child accessions, for now we just parse it all into a flat string
        # this should probably be improved in the future!! (but then we will simply upgrade to a different type that
        # can hold the extra information, so it should be obvious whether the new code is used)
        items = self._etree.findall(f".//{self._ns}processingMethod/{self._ns}cvParam")
        return [item.attrib["name"] for item in items]

    @property
    def software(self) -> list[str]:
        items = self._etree.findall(f".//{self._ns}software/{self._ns}cvParam")
        return [item.attrib["name"] for item in items]
