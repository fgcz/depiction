from __future__ import annotations

from dataclasses import dataclass
from xml.etree import ElementTree


@dataclass
class CvParam:
    accession: str
    name: str
    value: str


def extract_cv_param_list(els: list[ElementTree.Element]) -> list[CvParam]:
    return [
        CvParam(
            accession=el.attrib["accession"],
            name=el.attrib["name"],
            value=el.attrib.get("value", ""),
        )
        for el in els
    ]
