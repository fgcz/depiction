from __future__ import annotations

import pytest
from pytest_mock import MockerFixture

from depiction_targeted_preproc.workflow.snakemake_invoke import SnakemakeInvoke


@pytest.fixture
def default_invoke() -> SnakemakeInvoke:
    return SnakemakeInvoke()


def test_get_base_command(mocker: MockerFixture, default_invoke: SnakemakeInvoke) -> None:
    mock_shutil_which = mocker.patch("shutil.which", return_value="/custom/bin/snakemake")
    mocker.patch.object(SnakemakeInvoke, "snakefile_path", "/tmp/workflow/Snakefile")
    base_command = default_invoke.get_base_command(
        extra_args=["--keep-going"],
        work_dir="/tmp/work",
    )
    assert base_command == [
        "/custom/bin/snakemake",
        "-d",
        "/tmp/work",
        "--cores",
        "1",
        "--snakefile",
        "/tmp/workflow/Snakefile",
        "--rerun-incomplete",
        "--keep-going",
    ]
    mock_shutil_which.assert_called_once_with("snakemake")


def test_get_base_command_when_snakemake_not_found(mocker: MockerFixture, default_invoke: SnakemakeInvoke) -> None:
    mocker.patch("shutil.which", return_value=None)
    with pytest.raises(RuntimeError) as excinfo:
        default_invoke.get_base_command(
            extra_args=[],
            work_dir="/tmp/work",
        )
    assert str(excinfo.value).startswith("snakemake not found, check PATH:")


if __name__ == "__main__":
    pytest.main()
