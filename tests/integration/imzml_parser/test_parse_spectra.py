from pathlib import Path
from xml.etree import ElementTree

import pytest

from depiction.persistence.imzml.parser.parse_spectra import _CvParam


@pytest.fixture
def etree_referenceable_param_groups():
    path = Path(__file__).parent / "chunks" / "referenceable_param_groups.xml"
    return ElementTree.parse(path)


def test_referenceable_param_groups(etree_referenceable_param_groups):
    from depiction.persistence.imzml.parser.parse_spectra import ParseSpectra

    parser = ParseSpectra(etree_referenceable_param_groups)
    referenceable_param_groups = parser._referenceable_param_groups

    assert set(referenceable_param_groups.keys()) == {"mzArray", "intensityArray", "scan1", "spectrum1"}
    assert referenceable_param_groups["mzArray"] == [
        _CvParam(accession="MS:1000574", name="zlib compression", value=""),
        _CvParam(accession="MS:1000514", name="m/z array", value=""),
        _CvParam(accession="MS:1000523", name="64-bit float", value=""),
        _CvParam(accession="IMS:1000101", name="external data", value="true"),
    ]
    assert referenceable_param_groups["intensityArray"] == [
        _CvParam(accession="MS:1000523", name="64-bit float", value=""),
        _CvParam(accession="MS:1000515", name="intensity array", value=""),
        _CvParam(accession="MS:1000574", name="zlib compression", value=""),
        _CvParam(accession="IMS:1000101", name="external data", value="true"),
    ]
