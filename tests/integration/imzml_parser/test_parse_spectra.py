from pathlib import Path
from xml.etree import ElementTree

import pytest

from depiction.persistence.imzml.parser.parse_spectra import ParseSpectra
from depiction.persistence.imzml.parser.parse_spectra import _CvParam


@pytest.fixture
def etree_referenceable_param_groups():
    path = Path(__file__).parent / "chunks" / "referenceable_param_groups.xml"
    return ElementTree.parse(path)


@pytest.fixture
def etree_spectra_static():
    path = Path(__file__).parent / "chunks" / "spectra_static.xml"
    return ElementTree.parse(path)


def test_referenceable_param_groups(etree_referenceable_param_groups):
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


def test_spectra_static(etree_spectra_static):
    parser = ParseSpectra(etree_spectra_static)
    spectra_static = parser._spectra_static
    assert len(spectra_static) == 3
    assert spectra_static[0].id == "spectrum=1"
    assert spectra_static[0].index == 1
    assert spectra_static[0].param_groups == ["spectrum1"]
    assert spectra_static[0].param_groups_scan == ["scan1"]
    assert spectra_static[0].params == [
        _CvParam(accession="MS:1000528", name="lowest observed m/z", value="803.150729499946"),
        _CvParam(accession="MS:1000527", name="highest observed m/z", value="1721.4524145370692"),
        _CvParam(accession="MS:1000504", name="base peak m/z", value="1482.7548111416522"),
        _CvParam(accession="MS:1000505", name="base peak intensity", value="3667.0"),
        _CvParam(accession="MS:1000285", name="total ion current", value="23902.0"),
    ]
    assert spectra_static[0].params_scan == [
        _CvParam(accession="IMS:1000050", name="position x", value="2445"),
        _CvParam(accession="IMS:1000051", name="position y", value="822"),
    ]
    assert spectra_static[0].position == (2445, 822)
    assert len(spectra_static[0].binary_arrays) == 2

    assert spectra_static[0].binary_arrays[0].array_length == 82
    assert spectra_static[0].binary_arrays[0].encoded_length == 667
    assert spectra_static[0].binary_arrays[0].offset == 16
    assert spectra_static[0].binary_arrays[0].param_groups == ["mzArray"]

    assert spectra_static[0].binary_arrays[1].param_groups == ["intensityArray"]
