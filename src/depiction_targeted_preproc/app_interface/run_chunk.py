from pathlib import Path

import cyclopts
import yaml

from depiction_targeted_preproc.pipeline.run_workflow import run_workflow

app = cyclopts.App()


@app.default()
def run_chunk(chunk_dir: Path):
    zip_file_path = run_workflow(sample_dir=chunk_dir)
    results = [
        {
            "type": "bfabric_copy_resource",
            "local_path": str(zip_file_path.absolute()),
            "store_entry_path": zip_file_path.name,
        }
    ]
    with (chunk_dir / "outputs.yml").open("w") as f:
        yaml.safe_dump(results, f)


if __name__ == "__main__":
    app()
