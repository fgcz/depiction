import zipfile
from pathlib import Path

import cyclopts
import yaml
from snakemake_invoke import SnakemakeInvoke
from snakemake_invoke.config import SnakemakeInvokeConfig

from depiction_targeted_preproc.pipeline.prepare_params import Params
from depiction_targeted_preproc.pipeline_config.artifacts_mapping import get_result_files_new

app = cyclopts.App()


@app.default()
def process_chunk(chunk_dir: Path) -> Path:
    chunk_dir = chunk_dir.absolute()

    # TODO to be refactored
    params = Params.model_validate(yaml.safe_load((chunk_dir / "params.yml").read_text()))
    result_files = get_result_files_new(requested_artifacts=params.requested_artifacts, sample_dir=chunk_dir)

    # invoke snakemake
    # TODO should we generate the report_file again? before it was broken due to jinja2 update
    snakemake_config = SnakemakeInvokeConfig(
        snakefile_path=Path(__file__).parents[1] / "workflow" / "Snakefile",
    )
    SnakemakeInvoke(snakemake_config).invoke(work_dir=chunk_dir.parent, result_files=result_files)

    # zip the results
    sample_name = chunk_dir.name
    output_dir = chunk_dir / "outputs"
    output_dir.mkdir(exist_ok=True)
    zip_file_path = output_dir / f"{sample_name}.zip"
    with zipfile.ZipFile(zip_file_path, "w") as zip_file:
        for result_file in result_files:
            if result_file.is_file():
                zip_entry_path = result_file.relative_to(chunk_dir.parent)
                zip_file.write(result_file, arcname=zip_entry_path)
            elif result_file.is_dir():
                for file_path in result_file.rglob("*"):
                    if file_path.is_file():
                        zip_entry_path = file_path.relative_to(chunk_dir.parent)
                        zip_file.write(file_path, arcname=zip_entry_path)
    return zip_file_path


if __name__ == "__main__":
    app()
