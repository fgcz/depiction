from __future__ import annotations

from pathlib import Path

import yaml
from bfabric import Bfabric
from bfabric.entities import Workunit
from bfabric.experimental.app_interface.output_registration import register_outputs


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


def write_outputs_spec(zip_file_path: Path, workunit: Workunit) -> Path:
    output_spec = _get_outputs_spec(zip_file_path=zip_file_path, workunit=workunit)
    outputs_yaml = zip_file_path.parent / f"{zip_file_path.stem}_outputs_spec.yml"
    with outputs_yaml.open("w") as file:
        yaml.safe_dump(output_spec, file)
    return outputs_yaml


def store_outputs(client: Bfabric, zip_file_path: Path, workunit_id: int, ssh_user: str | None):
    workunit = Workunit.find(id=workunit_id, client=client)
    outputs_yaml = write_outputs_spec(zip_file_path=zip_file_path, workunit=workunit)
    register_outputs(outputs_yaml=outputs_yaml, client=client, ssh_user=ssh_user)
