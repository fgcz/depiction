from __future__ import annotations

from pathlib import Path

import pytest

from snakemake_invoke import SnakemakeInvoke
from snakemake_invoke.config import ExecutionModel, SnakemakeInvokeConfig


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
def mock_work_dir(mocker):
    return mocker.MagicMock(name="work_dir", spec=Path)


@pytest.fixture
def mock_result_files(mocker):
    return [mocker.MagicMock(name=f"result_file[{i}]", spec=Path) for i in range(3)]


@pytest.fixture
def invoke(config):
    return SnakemakeInvoke(config=config)


@pytest.mark.parametrize(
    "config_execution_model", [ExecutionModel.SUBPROCESS], indirect=True
)
def test_invoke_subprocess(mocker, invoke, mock_work_dir, mock_result_files):
    mocked = mocker.patch("snakemake_invoke.invoke.invoke.InvokeSubprocess")
    invoke.invoke(work_dir=mock_work_dir, result_files=mock_result_files)
    mocked.assert_called_once_with(config=invoke.config)
    mocked.return_value.invoke.assert_called_once_with(
        work_dir=mock_work_dir, result_files=mock_result_files
    )


@pytest.mark.parametrize(
    "config_execution_model", [ExecutionModel.CALL_FUNCTION], indirect=True
)
def test_invoke_call_function(mocker, invoke, mock_work_dir, mock_result_files):
    mocked = mocker.patch("snakemake_invoke.invoke.invoke.InvokeCallFunction")
    invoke.invoke(work_dir=mock_work_dir, result_files=mock_result_files)
    mocked.assert_called_once_with(config=invoke.config)
    mocked.return_value.invoke.assert_called_once_with(
        work_dir=mock_work_dir, result_files=mock_result_files
    )


def test_invoke_when_unknown(invoke, mock_work_dir, mock_result_files):
    invoke.config.execution_model = "unknown"
    with pytest.raises(AssertionError) as error:
        invoke.invoke(work_dir=mock_work_dir, result_files=mock_result_files)
    assert "unreachable" in str(error.value)
