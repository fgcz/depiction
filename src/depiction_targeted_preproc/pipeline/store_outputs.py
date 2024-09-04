from __future__ import annotations

from pathlib import Path

import yaml
from bfabric import Bfabric
from bfabric.entities import Workunit


def _get_outputs_spec(zip_file_path: Path, workunit: Workunit) -> dict[str, list[dict[str, str | int | bool]]]:
    return {
        "outputs": [
            {
                "type": "bfabric_copy_resource",
                "local_path": str(zip_file_path.absolute()),
                "store_entry_path": zip_file_path.name,
                "workunit_id": workunit.id,
                "storage_id": workunit.application.storage.id,
            }
        ]
    }


def write_outputs_spec(zip_file_path: Path, workunit: Workunit) -> None:
    output_spec = _get_outputs_spec(zip_file_path=zip_file_path, workunit=workunit)
    outputs_yaml = zip_file_path.parent / f"{zip_file_path.stem}_outputs_spec.yml"
    with outputs_yaml.open("w") as file:
        yaml.safe_dump(output_spec, file)


def store_outputs(client: Bfabric, zip_file_path: Path, workunit_id: int):
    workunit = Workunit.find(id=workunit_id, client=client)
    write_outputs_spec(zip_file_path=zip_file_path, workunit=workunit)
