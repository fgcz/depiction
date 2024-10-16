from pathlib import Path
from xml.etree import ElementTree

import pytest

from depiction.persistence.imzml.compression import Compression
from depiction.persistence.imzml.parser.parse_spectra import ParseSpectra
from depiction.persistence.imzml.parser.cv_params import CvParam


@pytest.fixture
def etree_referenceable_param_groups():
    path = Path(__file__).parent / "chunks" / "referenceable_param_groups.xml"
    return ElementTree.parse(path)


@pytest.fixture
def etree_spectra_static():
    path = Path(__file__).parent / "chunks" / "spectra_static.xml"
    return ElementTree.parse(path)


@pytest.fixture
def etree_spectra_resolve():
    path = Path(__file__).parent / "chunks" / "spectra_resolve.xml"
    return ElementTree.parse(path)


def test_referenceable_param_groups(etree_referenceable_param_groups):
    parser = ParseSpectra(etree_referenceable_param_groups)
    referenceable_param_groups = parser._referenceable_param_groups

    assert set(referenceable_param_groups.keys()) == {"mzArray", "intensityArray", "scan1", "spectrum1"}
    assert referenceable_param_groups["mzArray"] == [
        CvParam(accession="MS:1000574", name="zlib compression", value=""),
        CvParam(accession="MS:1000514", name="m/z array", value=""),
        CvParam(accession="MS:1000523", name="64-bit float", value=""),
        CvParam(accession="IMS:1000101", name="external data", value="true"),
    ]
    assert referenceable_param_groups["intensityArray"] == [
        CvParam(accession="MS:1000523", name="64-bit float", value=""),
        CvParam(accession="MS:1000515", name="intensity array", value=""),
        CvParam(accession="MS:1000574", name="zlib compression", value=""),
        CvParam(accession="IMS:1000101", name="external data", value="true"),
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
        CvParam(accession="MS:1000528", name="lowest observed m/z", value="803.150729499946"),
        CvParam(accession="MS:1000527", name="highest observed m/z", value="1721.4524145370692"),
        CvParam(accession="MS:1000504", name="base peak m/z", value="1482.7548111416522"),
        CvParam(accession="MS:1000505", name="base peak intensity", value="3667.0"),
        CvParam(accession="MS:1000285", name="total ion current", value="23902.0"),
    ]
    assert spectra_static[0].params_scan == [
        CvParam(accession="IMS:1000050", name="position x", value="2445"),
        CvParam(accession="IMS:1000051", name="position y", value="822"),
    ]
    assert spectra_static[0].position == (2445, 822)
    assert len(spectra_static[0].binary_arrays) == 2

    assert spectra_static[0].binary_arrays[0].array_length == 82
    assert spectra_static[0].binary_arrays[0].encoded_length == 667
    assert spectra_static[0].binary_arrays[0].offset == 16
    assert spectra_static[0].binary_arrays[0].param_groups == ["mzArray"]

    assert spectra_static[0].binary_arrays[1].param_groups == ["intensityArray"]


def test_spectra_resolve(etree_spectra_resolve):
    parser = ParseSpectra(etree_spectra_resolve)
    resolved = parser._spectra_resolved
    assert len(resolved) == 3

    assert resolved[0].id == "spectrum=1"
    assert resolved[0].index == 1
    assert set(resolved[0].params.keys()) == {
        "MS:1000127",
        "MS:1000130",
        "MS:1000285",
        "MS:1000504",
        "MS:1000505",
        "MS:1000511",
        "MS:1000527",
        "MS:1000528",
        "MS:1000579",
    }
    # TODO check the list is actually what we want
    # TODO params_scan
    assert resolved[0].position == (2445, 822)
    assert resolved[0].mz_arr.array_length == 82
    assert resolved[0].mz_arr.encoded_length == 667
    assert resolved[0].mz_arr.offset == 16
    assert resolved[0].mz_arr.compression == Compression.Zlib
    assert resolved[0].int_arr.array_length == 82
    assert resolved[0].int_arr.encoded_length == 215
    assert resolved[0].int_arr.offset == 683
    assert resolved[0].int_arr.compression == Compression.Zlib


def test_parse(etree_spectra_resolve):
    parser = ParseSpectra(etree_spectra_resolve)
    spectra = parser.parse()
    assert len(spectra) == 3
    assert spectra[0].position == (2445, 822)
    assert spectra[0].mz_arr.encoded_length == 667
    assert spectra[0].int_arr.encoded_length == 215
