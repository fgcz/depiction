import warnings

import pytest


@pytest.fixture()
def treat_warnings_as_error():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        yield
