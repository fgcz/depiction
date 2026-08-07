"""Runs one fixture through `depiction_targeted_preproc` in two trees, for comparison.

Both work directories are staged from here, by the *same* code, so that any difference in
the outputs comes from the pipeline rather than from how it was set up. Only the
interpreter differs: this tree's, and the pre-refactor baseline's.

`process_chunk` is invoked as a subprocess rather than imported, because the snakemake rules
it dispatches are `python -m depiction...` shell commands that resolve against `PATH`. Both
the parent and its rule processes therefore have to see the right environment, which an
in-process call could not arrange.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import cyclopts

from system_tests.fixtures import FIXTURES_BY_NAME, PipelineFixture

app = cyclopts.App()

_CONFIG_DIR = Path(__file__).parents[1] / "calibration" / "configs"


def stage(fixture: PipelineFixture, work_dir: Path) -> None:
    """Lays out a chunk directory exactly as `system_tests/calibration/` does.

    Kept in step with the system test's `work_dir` fixture deliberately: the point of the
    comparison is the pipeline the system tests already cover, not a second configuration.
    """
    (work_dir / "panels").mkdir(parents=True)
    shutil.copy(fixture.imzml_path, work_dir / "raw.imzML")
    shutil.copy(fixture.ibd_path, work_dir / "raw.ibd")
    shutil.copy(fixture.panel_path, work_dir / "panels" / "unstandardized_full.csv")
    shutil.copy(_CONFIG_DIR / "params.yml", work_dir / "params.yml")
    shutil.copy(_CONFIG_DIR / "global_constant_shift.yml", work_dir / "pipeline_params.yml")


def run_pipeline(python: Path, work_dir: Path) -> None:
    """Runs `process_chunk` in the environment `python` belongs to."""
    venv_bin = python.parent
    env = {
        **os.environ,
        "PATH": f"{venv_bin}{os.pathsep}{os.environ['PATH']}",
        "VIRTUAL_ENV": str(venv_bin.parent),
    }
    # An inherited PYTHONPATH would let one tree's sources leak into the other's run, which
    # is the one contamination this whole comparison cannot tolerate.
    env.pop("PYTHONPATH", None)
    subprocess.run(
        [str(python), "-m", "depiction_targeted_preproc.app_interface.process_chunk", str(work_dir)],
        check=True,
        env=env,
        cwd=venv_bin.parents[1],
    )


def record_versions(python: Path, target: Path) -> None:
    """Freezes the environment next to its outputs, so the report cannot misquote it."""
    frozen = subprocess.run(
        ["uv", "pip", "freeze", "--python", str(python)], check=True, capture_output=True, text=True
    )
    target.write_text(frozen.stdout)


@app.default()
def main(
    *,
    fixture_name: str,
    baseline_python: Path,
    out_dir: Path,
    current_python: Path = Path(sys.executable),
) -> None:
    """Stages and runs `fixture_name` in both trees under `out_dir`."""
    fixture = FIXTURES_BY_NAME[fixture_name]
    if fixture.missing:
        raise SystemExit(fixture.skip_reason)

    for tree, python in (("baseline", baseline_python), ("current", current_python)):
        work_dir = out_dir / fixture.name / tree / "work"
        if work_dir.exists():
            raise SystemExit(f"{work_dir} already exists; remove it or choose another --out-dir")
        work_dir.mkdir(parents=True)
        stage(fixture, work_dir)
        record_versions(python, work_dir.parent / "versions.txt")
        print(f"--- {fixture.name}: running the {tree} tree in {work_dir}", flush=True)
        run_pipeline(python, work_dir)


if __name__ == "__main__":
    app()
