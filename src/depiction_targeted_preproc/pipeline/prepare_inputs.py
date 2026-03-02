from __future__ import annotations

from pathlib import Path

import yaml
from bfabric import Bfabric
from bfabric.entities import Resource


def _get_imzml_inputs(imzml_resource: Resource, client: Bfabric) -> list[dict[str, str | int | bool]]:
    expected_name = imzml_resource["name"][:-6] + ".ibd"
    results = client.read(
        "resource",
        {"name": expected_name, "containerid": imzml_resource["container"]["id"]},
        max_results=1,
        return_id_only=True,
    )
    ibd_resource_id = results[0]["id"]

    return [
        {
            "type": "bfabric_resource",
            "id": imzml_resource.id,
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


def _get_imzml_zip_inputs(imzml_resource: Resource) -> list[dict[str, str | int | bool]]:
    # Workaround, the extraction is handled in Snakemake prepare_pipeline_extract_imzml_zip
    return [{"type": "bfabric_resource", "id": imzml_resource.id, "filename": "raw.imzML.zip"}]


def _get_inputs_spec(
    dataset_id: int, imzml_resource_id: int, client: Bfabric
) -> dict[str, list[dict[str, str | int | bool]]]:
    inputs = [
        {
            "type": "bfabric_dataset",
            "id": dataset_id,
            "filename": "panels/unstandardized_full.csv",
            "separator": ",",
        }
    ]

    imzml_resource = Resource.find(id=imzml_resource_id, client=client)
    if imzml_resource["name"].endswith(".imzML"):
        inputs.extend(_get_imzml_inputs(imzml_resource=imzml_resource))
    elif imzml_resource["name"].lower().endswith("imzml.zip"):
        # NOTE: we cannot check .imzml.zip because it could be _imzml.zip as well due to data placement rules
        inputs.extend(_get_imzml_zip_inputs(imzml_resource=imzml_resource))
    else:
        msg = f"Unsupported imzml resource encountered: {imzml_resource['name']}"
        raise NotImplementedError(msg)

    return {"inputs": inputs}


def write_inputs_spec(dataset_id: int, imzml_resource_id: int, client: Bfabric, sample_dir: Path) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    inputs_spec = _get_inputs_spec(dataset_id=dataset_id, imzml_resource_id=imzml_resource_id, client=client)
    inputs_yaml = sample_dir / "inputs.yml"
    with inputs_yaml.open("w") as file:
        yaml.safe_dump(inputs_spec, file)
