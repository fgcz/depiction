from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from bfabric import Bfabric
from bfabric.entities import Resource, Workunit
from bfabric.experimental.app_interface.input_preparation import prepare_folder


def _get_ibd_resource_id(imzml_resource_id: int, client: Bfabric) -> int:
    imzml_resource = Resource.find(id=imzml_resource_id, client=client)
    if imzml_resource["name"].endswith(".imzML"):
        expected_name = imzml_resource["name"][:-6] + ".ibd"
        results = client.read(
            "resource",
            {"name": expected_name, "containerid": imzml_resource["container"]["id"]},
            max_results=1,
            return_id_only=True,
        )
        return results[0]["id"]
    else:
        # TODO this will have to be refactored later
        raise NotImplementedError("Only .imzML files are supported for now")


def _get_inputs_spec(
    dataset_id: int, imzml_resource_id: int, client: Bfabric
) -> dict[str, list[dict[str, str | int | bool]]]:
    ibd_resource_id = _get_ibd_resource_id(imzml_resource_id=imzml_resource_id, client=client)
    return {
        "inputs": [
            {
                "type": "bfabric_dataset",
                "id": dataset_id,
                "filename": "mass_list.unstandardized.raw.csv",
                "separator": ",",
            },
            {
                "type": "bfabric_resource",
                "id": imzml_resource_id,
                "filename": "raw.imzML",
                "check_checksum": True,
            },
            {
                "type": "bfabric_resource",
                "id": ibd_resource_id,
                "filename": "raw.ibd",
                "check_checksum": True,
            },
        ]
    }


def write_inputs_spec(dataset_id: int, imzml_resource_id: int, client: Bfabric, sample_dir: Path) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    inputs_spec = _get_inputs_spec(dataset_id=dataset_id, imzml_resource_id=imzml_resource_id, client=client)
    inputs_yaml = sample_dir / "inputs.yml"
    with inputs_yaml.open("w") as file:
        yaml.safe_dump(inputs_spec, file)


def prepare_inputs(
    client: Bfabric,
    sample_dir: Path,
    workunit_id: int,
    ssh_user: str | None,
) -> None:
    dataset_id, imzml_resource_id = _get_resource_flow_ids(client=client, workunit_id=workunit_id)
    write_inputs_spec(dataset_id=dataset_id, imzml_resource_id=imzml_resource_id, client=client, sample_dir=sample_dir)
    prepare_folder(inputs_yaml=sample_dir / "inputs.yml", target_folder=sample_dir, client=client, ssh_user=ssh_user)


def _get_resource_flow_ids(client: Bfabric, workunit_id: int) -> tuple[int, int]:
    workunit = Workunit.find(id=workunit_id, client=client)
    dataset_id = workunit.parameter_values["mass_list_id"]
    imzml_resources = [r for r in workunit.input_resources if r["name"].endswith(".imzML")]
    if len(imzml_resources) != 1:
        raise ValueError(f"Expected exactly one .imzML resource, found {len(imzml_resources)}")
    imzml_resource_id = imzml_resources[0].id
    return dataset_id, imzml_resource_id