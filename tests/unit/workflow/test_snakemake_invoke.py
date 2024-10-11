from __future__ import annotations

from pathlib import Path

import pytest
import sys
from pytest_mock import MockerFixture

from depiction_targeted_preproc.workflow.snakemake_invoke import SnakemakeInvoke


@pytest.fixture
def default_invoke() -> SnakemakeInvoke:
    return SnakemakeInvoke()


def test_get_base_command(mocker: MockerFixture, default_invoke: SnakemakeInvoke) -> None:
    mocker.patch.object(SnakemakeInvoke, "snakefile_path", "/tmp/workflow/Snakefile")
    base_command = default_invoke.get_base_command(
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


if __name__ == "__main__":
    pytest.main()
