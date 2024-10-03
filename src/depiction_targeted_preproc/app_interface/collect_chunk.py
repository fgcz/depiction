from pathlib import Path

import cyclopts
import yaml

from bfabric import Bfabric
from bfabric.experimental.app_interface.workunit.definition import WorkunitDefinition

app = cyclopts.App()


@app.default
def collect_chunk(workunit_ref: int | Path, chunk_dir: Path) -> None:
    chunk_dir = chunk_dir.absolute()
    zip_file_path = chunk_dir / f"{chunk_dir.name}.zip"

    # TODO how to incorporate the cache_file parameter here, without "assuming" that it's in the parent directory?
    workunit_definition = WorkunitDefinition.from_ref(workunit_ref, client=Bfabric.from_config())
    workunit_id = workunit_definition.registration.workunit_id

    outputs = [
        {
            "type": "bfabric_copy_resource",
            "local_path": str(zip_file_path.absolute()),
            "store_entry_path": f"WU{workunit_id}_result_{chunk_dir.name}.zip",
        }
    ]

    with (chunk_dir / "outputs.yml").open("w") as f:
        yaml.safe_dump({"outputs": outputs}, f)


if __name__ == "__main__":
    app()
