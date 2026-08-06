from __future__ import annotations

import pytest

from tests.differential.corpus import CASE_NAMES, Case, build_corpus


@pytest.fixture(scope="session")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Case]:
    """The differential corpus, built once per session and keyed by case name."""
    cases = build_corpus(tmp_path_factory.mktemp("differential_corpus"))
    return {case.name: case for case in cases}


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrises over corpus case *names*.

    The files live in a session fixture, so tests take a name and look it up rather than
    parametrising over the objects -- otherwise the whole corpus would have to be written
    during collection.
    """
    if "case_name" in metafunc.fixturenames:
        names = CASE_NAMES
        if metafunc.definition.get_closest_marker("compressed_only"):
            names = [name for name in names if name.endswith("_zlib")]
        metafunc.parametrize("case_name", names)


@pytest.fixture
def case(case_name: str, corpus: dict[str, Case]) -> Case:
    return corpus[case_name]
