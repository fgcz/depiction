from __future__ import annotations

import platform
import sys
from pathlib import Path

import pytest

from snakemake_invoke.config import SnakemakeInvokeConfig
from snakemake_invoke.invoke.invoke_subprocess import InvokeSubprocess


@pytest.fixture
def config_snakefile_path() -> Path:
    return Path("/tmp/workflow/Snakefile")


@pytest.fixture
def config(config_snakefile_path):
    return SnakemakeInvokeConfig(snakefile_path=config_snakefile_path)


@pytest.fixture
def invoke(config):
    return InvokeSubprocess(config=config)


@pytest.mark.parametrize(
    ("input_args", "expected"),
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
    result = InvokeSubprocess._args_to_shell_command(input_args)
    assert result == expected


@pytest.mark.skipif(platform.system() == "Windows", reason="requires unix")
def test_get_base_command(invoke) -> None:
    base_command = invoke.get_base_command(
        extra_args=["--keep-going"],
        work_dir=Path("/tmp/work"),
    )
    assert base_command == [
        sys.executable,
        "-m",
        "snakemake",
        "-d",
        "/tmp/work",
        "--cores",
        "1",
        "--snakefile",
        "/tmp/workflow/Snakefile",
        "--rerun-incomplete",
        "--keep-going",
    ]
