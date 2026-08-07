from __future__ import annotations

import os

import pytest

from snakemake_invoke.config import SnakemakeInvokeConfig
from snakemake_invoke.invoke.invoke_call_function import InvokeCallFunction


@pytest.fixture(params=[None])
def config_env_variables(request) -> dict[str, str]:
    return request.param or {}


@pytest.fixture
def config(config_env_variables):
    return SnakemakeInvokeConfig(
        env_variables=config_env_variables,
    )


@pytest.fixture
def invoke(config):
    return InvokeCallFunction(config=config)


@pytest.mark.parametrize(
    ("config_env_variables", "expected"),
    [
        ({}, {"a": "1", "b": "2"}),
        ({"c": "3"}, {"a": "1", "b": "2", "c": "3"}),
        ({"a": "x", "c": "3"}, {"a": "x", "b": "2", "c": "3"}),
    ],
    indirect=["config_env_variables"],
)
def test_set_env_vars(mocker, invoke, expected):
    original_env_vars = {"a": "1", "b": "2"}
    mocker.patch("os.environ", original_env_vars)
    with invoke._set_env_vars():
        assert dict(os.environ) == expected
    assert dict(os.environ) == {"a": "1", "b": "2"}
