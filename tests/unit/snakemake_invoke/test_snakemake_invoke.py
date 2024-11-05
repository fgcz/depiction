import os

import pytest
from snakemake_invoke.config import SnakemakeInvokeConfig, ExecutionModel
from snakemake_invoke.snakemake_invoke import SnakemakeInvoke


@pytest.fixture(params=[None])
def config_execution_model(request) -> ExecutionModel:
    return request.param or ExecutionModel.SUBPROCESS


@pytest.fixture(params=[None])
def config_env_variables(request) -> dict[str, str]:
    return request.param or {}


@pytest.fixture
def config(config_execution_model, config_env_variables):
    return SnakemakeInvokeConfig(
        execution_model=config_execution_model,
        env_variables=config_env_variables,
    )


@pytest.fixture
def invoke(config):
    return SnakemakeInvoke(config=config)


@pytest.mark.parametrize(
    "config_execution_model,method",
    [
        (ExecutionModel.SUBPROCESS, "_invoke_subprocess"),
        (ExecutionModel.CALL_FUNCTION, "_invoke_direct"),
    ],
    indirect=["config_execution_model"],
)
def test_invoke(mocker, invoke, method):
    mock_work_dir = mocker.MagicMock(name="work_dir", spec=[])
    mock_result_files = mocker.MagicMock(name="result_files", spec=[])
    mocked_method = mocker.patch.object(invoke, method)
    invoke.invoke(work_dir=mock_work_dir, result_files=mock_result_files)
    mocked_method.assert_called_once_with(mock_work_dir, mock_result_files)


def test_invoke_when_unknown(mocker, invoke):
    mock_work_dir = mocker.MagicMock(name="work_dir", spec=[])
    mock_result_files = mocker.MagicMock(name="result_files", spec=[])
    invoke.config.execution_model = "unknown"
    with pytest.raises(ValueError) as error:
        invoke.invoke(work_dir=mock_work_dir, result_files=mock_result_files)
    assert str(error.value) == f"Unknown execution model: unknown"


@pytest.mark.parametrize(
    "config_env_variables,expected",
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


@pytest.mark.parametrize(
    "input_args,expected",
    [
        pytest.param(["a"], "a", id="plain_string"),
        pytest.param(["a b"], "'a b'", id="string_with_space"),
        pytest.param(['"a b"'], "'\"a b\"'", id="string_with_double_quotes"),
        pytest.param(["a'b"], "'a'\"'\"'b'", id="string_with_single_quote"),
        pytest.param(['a"b'], "'a\"b'", id="string_with_double_quote"),
        pytest.param(["a", "a b", ""], "a 'a b' ''", id="list_of_strings"),
    ],
)
def test_args_to_shell_command(input_args, expected):
    result = SnakemakeInvoke._args_to_shell_command(input_args)
    assert result == expected
