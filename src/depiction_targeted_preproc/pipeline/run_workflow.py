import zipfile
from pathlib import Path

import cyclopts
import yaml
from depiction_targeted_preproc.pipeline.prepare_params import Params
from depiction_targeted_preproc.pipeline_config.artifacts_mapping import get_result_files_new
from depiction_targeted_preproc.workflow.snakemake_invoke import SnakemakeInvoke


app = cyclopts.App()


@app.default
def run_workflow(sample_dir: Path) -> Path:
    # TODO to be refactored
    params = Params.model_validate(yaml.safe_load((sample_dir / "params.yml").read_text()))
    result_files = get_result_files_new(requested_artifacts=params.requested_artifacts, sample_dir=sample_dir)

    # invoke snakemake
    # TODO note report file is deactivated because it's currently broken due to dependencies (jinja2)
    SnakemakeInvoke(report_file=None).invoke(work_dir=sample_dir.parent, result_files=result_files)

    # zip the results
    sample_name = sample_dir.name
    output_dir = sample_dir.parent / "output"
    output_dir.mkdir(exist_ok=True)
    zip_file_path = output_dir / f"{sample_name}.zip"
    with zipfile.ZipFile(zip_file_path, "w") as zip_file:
        for result_file in result_files:
            zip_entry_path = result_file.relative_to(sample_dir.parent)
            zip_file.write(result_file, arcname=zip_entry_path)
    return zip_file_path


if __name__ == "__main__":
    app()
