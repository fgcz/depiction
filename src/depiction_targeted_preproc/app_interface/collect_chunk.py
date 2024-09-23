from pathlib import Path

import cyclopts
import yaml

app = cyclopts.App()


@app.default
def collect_chunk(workunit_ref: int | Path, chunk_dir: Path) -> None:
    chunk_dir = chunk_dir.absolute()
    zip_file_path = chunk_dir / f"{chunk_dir.name}.zip"

    outputs = [
        {
            "type": "bfabric_copy_resource",
            "local_path": str(zip_file_path.absolute()),
            "store_entry_path": zip_file_path.name,
        }
    ]
    result = {"outputs": outputs}
    with (chunk_dir / "outputs.yml").open("w") as f:
        yaml.safe_dump(result, f)


if __name__ == "__main__":
    app()
