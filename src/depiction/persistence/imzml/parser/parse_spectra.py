from dataclasses import dataclass
from xml.etree.ElementTree import ElementTree


@dataclass
class _CvParam:
    accession: str
    name: str
    value: str


@dataclass
class _BinaryArray:
    array_length: int
    encoded_length: int
    offset: int
    param_groups: list[str]


@dataclass
class _SpectrumStatic:
    id: str
    index: int
    param_groups: list[str]
    param_groups_scan: list[str]
    params: list[_CvParam]
    params_scan: list[_CvParam]
    position: tuple[int, int] | tuple[int, int, int]
    binary_arrays: list[_BinaryArray]


class ParseSpectra:
    def __init__(self, etree: ElementTree):
        self._etree = etree
        self._ns = "{http://psi.hupo.org/ms/mzml}"

    @property
    def _referenceable_param_groups(self) -> dict[str, list[_CvParam]]:
        elements = self._etree.findall(f"/{self._ns}referenceableParamGroupList/{self._ns}referenceableParamGroup")
        result = {}
        for element in elements:
            result[element.attrib["id"]] = self._extract_cv_param_element_list(element.findall(f"{self._ns}cvParam"))
        return result

    @property
    def _spectra_static(self) -> list[_SpectrumStatic]:
        # TODO multi run support...
        elements = self._etree.findall(f"/{self._ns}run/{self._ns}spectrumList/{self._ns}spectrum")
        results = []
        for element in elements:
            param_groups = [ref.attrib["ref"] for ref in element.findall(f"{self._ns}referenceableParamGroupRef")]
            param_groups_scan = [
                ref.attrib["ref"]
                for ref in element.findall(f"{self._ns}scanList/{self._ns}scan/{self._ns}referenceableParamGroupRef")
            ]

            position_x = element.find(f"{self._ns}scanList/{self._ns}scan/{self._ns}cvParam[@accession='IMS:1000050']")
            position_y = element.find(f"{self._ns}scanList/{self._ns}scan/{self._ns}cvParam[@accession='IMS:1000051']")
            position_z = element.find(f"{self._ns}scanList/{self._ns}scan/{self._ns}cvParam[@accession='IMS:1000052']")

            if not position_z:
                position = (int(position_x.attrib["value"]), int(position_y.attrib["value"]))
            else:
                position = (
                    int(position_x.attrib["value"]),
                    int(position_y.attrib["value"]),
                    int(position_z.attrib["value"]),
                )

            binary_arrays = []
            for binary_element in element.findall(f"{self._ns}binaryDataArrayList/{self._ns}binaryDataArray"):
                binary_arrays.append(
                    _BinaryArray(
                        array_length=int(
                            binary_element.find(f"{self._ns}cvParam[@accession='IMS:1000103']").attrib["value"]
                        ),
                        encoded_length=int(
                            binary_element.find(f"{self._ns}cvParam[@accession='IMS:1000104']").attrib["value"]
                        ),
                        offset=int(binary_element.find(f"{self._ns}cvParam[@accession='IMS:1000102']").attrib["value"]),
                        param_groups=[
                            ref.attrib["ref"] for ref in binary_element.findall(f"{self._ns}referenceableParamGroupRef")
                        ],
                    )
                )

            results.append(
                _SpectrumStatic(
                    id=element.attrib["id"],
                    index=int(element.attrib["index"]),
                    param_groups=param_groups,
                    param_groups_scan=param_groups_scan,
                    params=self._extract_cv_param_element_list(element.findall(f"{self._ns}cvParam")),
                    params_scan=self._extract_cv_param_element_list(
                        element.findall(f"{self._ns}scanList/{self._ns}scan/{self._ns}cvParam")
                    ),
                    position=position,
                    binary_arrays=binary_arrays,
                )
            )

        return results

    @staticmethod
    def _extract_cv_param_element_list(els):
        return [
            _CvParam(
                accession=el.attrib["accession"],
                name=el.attrib["name"],
                value=el.attrib.get("value", ""),
            )
            for el in els
        ]
