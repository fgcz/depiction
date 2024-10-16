from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from xml.etree.ElementTree import ElementTree

import cyclopts
from loguru import logger

from depiction.persistence.imzml.compression import Compression


class DataType(str, Enum):
    Float64 = "float64"
    Float32 = "float32"


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
class _ResolvedBinaryArray:
    array_length: int
    encoded_length: int
    offset: int
    params: dict[str, _CvParam]
    compression: Compression
    data_type: DataType


@dataclass
class _ParsedBinaryArrayMinimal:
    array_length: int
    encoded_length: int
    offset: int
    compression: Compression
    data_type: DataType

    @classmethod
    def from_resolved(cls, b: _ResolvedBinaryArray) -> _ParsedBinaryArrayMinimal:
        return cls(
            array_length=b.array_length,
            encoded_length=b.encoded_length,
            offset=b.offset,
            compression=b.compression,
            data_type=b.data_type,
        )


@dataclass
class _ResolvedSpectrum:
    id: str
    index: int
    params: dict[str, _CvParam]
    params_scan: dict[str, _CvParam]
    position: tuple[int, int] | tuple[int, int, int]
    mz_arr: _ResolvedBinaryArray
    int_arr: _ResolvedBinaryArray


@dataclass
class _ParsedSpectrumMinimal:
    mz_arr: _ParsedBinaryArrayMinimal
    int_arr: _ParsedBinaryArrayMinimal
    position: tuple[int, int] | tuple[int, int, int]

    @classmethod
    def from_resolved(cls, s: _ResolvedSpectrum) -> _ParsedSpectrumMinimal:
        return cls(
            mz_arr=_ParsedBinaryArrayMinimal.from_resolved(s.mz_arr),
            int_arr=_ParsedBinaryArrayMinimal.from_resolved(s.int_arr),
            position=s.position,
        )


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

    def resolve_mz_int_array(
        self, groups: dict[str, list[_CvParam]]
    ) -> tuple[_ResolvedBinaryArray, _ResolvedBinaryArray]:
        mz_array = None
        int_array = None
        for binary_array in self.binary_arrays:
            params = _resolve_params(
                groups=groups, referenced_groups=self.param_groups + binary_array.param_groups, params=self.params
            )
            if "MS:1000514" in params or "MS:1000515" in params:
                if "MS:1000574" in params:
                    compression = Compression.Zlib
                else:
                    if "MS:1000576" not in params:
                        logger.error(f"Unknown compression for {self.id}, setting to uncompressed")
                    compression = Compression.Uncompressed

                if "MS:1000523" in params:
                    data_type = DataType.Float64
                else:
                    if "MS:1000521" not in params:
                        logger.error(f"Unknown data type for {self.id}, setting to float32")
                    data_type = DataType.Float32

                resolved = _ResolvedBinaryArray(
                    array_length=binary_array.array_length,
                    encoded_length=binary_array.encoded_length,
                    offset=binary_array.offset,
                    params=params,
                    compression=compression,
                    data_type=data_type,
                )
                if "MS:1000514" in params:
                    mz_array = resolved
                else:
                    int_array = resolved
        if not mz_array or not int_array:
            # TODO
            raise ValueError()
        return mz_array, int_array

    def resolve(self, groups: dict[str, list[_CvParam]]) -> _ResolvedSpectrum:
        mz_array, int_array = self.resolve_mz_int_array(groups)
        return _ResolvedSpectrum(
            id=self.id,
            index=self.index,
            params=_resolve_params(groups=groups, referenced_groups=self.param_groups, params=self.params),
            params_scan=_resolve_params(
                groups=groups, referenced_groups=self.param_groups_scan, params=self.params_scan
            ),
            position=self.position,
            mz_arr=mz_array,
            int_arr=int_array,
        )


def _resolve_params(
    groups: dict[str, list[_CvParam]], referenced_groups: list[str], params: list[_CvParam]
) -> dict[str, _CvParam]:
    resolved = {param.accession: param for param in params}
    for group in referenced_groups:
        for param in groups[group]:
            resolved[param.accession] = param
    return resolved


class ParseSpectra:
    def __init__(self, etree: ElementTree):
        self._etree = etree
        self._ns = "{http://psi.hupo.org/ms/mzml}"

    def parse(self) -> list[_ParsedSpectrumMinimal]:
        return [_ParsedSpectrumMinimal.from_resolved(spectrum) for spectrum in self._spectra_resolved]

    @property
    def _referenceable_param_groups(self) -> dict[str, list[_CvParam]]:
        elements = self._etree.findall(f"./{self._ns}referenceableParamGroupList/{self._ns}referenceableParamGroup")
        result = {}
        for element in elements:
            result[element.attrib["id"]] = self._extract_cv_param_element_list(element.findall(f"{self._ns}cvParam"))
        return result

    @property
    def _spectra_static(self) -> list[_SpectrumStatic]:
        # TODO multi run support...
        elements = self._etree.findall(f"./{self._ns}run/{self._ns}spectrumList/{self._ns}spectrum")
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

    @property
    def _spectra_resolved(self) -> list[_ResolvedSpectrum]:
        groups = self._referenceable_param_groups
        return [spectrum.resolve(groups) for spectrum in self._spectra_static]

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


# TODO maybe remove after testing?

app = cyclopts.App()


@app.command(name="print")
def print_(imzml_path: Path, pretty: bool = False):
    etree = ElementTree(file=imzml_path)
    parser = ParseSpectra(etree)
    print("parsing...")
    parsed = parser.parse()
    print("formatting...")
    if pretty:
        from rich.pretty import pprint

        pprint(parsed)
    else:
        print(json.dumps([dataclasses.asdict(d) for d in parsed], indent=2))


if __name__ == "__main__":
    app()
