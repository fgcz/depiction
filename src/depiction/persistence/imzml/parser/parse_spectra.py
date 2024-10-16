from dataclasses import dataclass
from xml.etree.ElementTree import ElementTree


@dataclass
class _CvParam:
    accession: str
    name: str
    value: str


class ParseSpectra:
    def __init__(self, etree: ElementTree):
        self._etree = etree
        self._ns = "{http://psi.hupo.org/ms/mzml}"

    @property
    def _referenceable_param_groups(self) -> dict[str, list[_CvParam]]:
        elements = self._etree.findall(f"/{self._ns}referenceableParamGroupList/{self._ns}referenceableParamGroup")
        result = {}
        for element in elements:
            result[element.attrib["id"]] = [
                _CvParam(
                    accession=cv_param.attrib["accession"],
                    name=cv_param.attrib["name"],
                    value=cv_param.attrib.get("value", ""),
                )
                for cv_param in element.findall(f"{self._ns}cvParam")
            ]
        return result
