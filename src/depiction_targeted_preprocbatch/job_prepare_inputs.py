from __future__ import annotations

from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from bfabric import Bfabric
from bfabric.entities import Resource
from bfabric.experimental.app_interface.input_preparation import prepare_folder
from depiction_targeted_preproc.pipeline.setup_old import copy_standardized_table
from loguru import logger

if TYPE_CHECKING:
    from depiction_targeted_preprocbatch.executor import BatchJob


class JobPrepareInputs:
    """Prepares the inputs for a particular job.
    :param job: The job to prepare the inputs for.
    :param sample_dir: The directory where the inputs should be staged.
    """

    def __init__(self, job: BatchJob, sample_dir: Path, client: Bfabric) -> None:
        self._job = job
        self._sample_dir = sample_dir
        self._client = client
        self._dataset_id = job.dataset_id
        self._imzml_resource_id = job.imzml_resource_id
        self._ssh_user = job.ssh_user

    @classmethod
    def prepare(cls, job: BatchJob, sample_dir: Path, client: Bfabric) -> None:
        """Prepares the inputs for a particular job.
        :param job: The job to prepare the inputs for.
        :param sample_dir: The directory where the inputs should be staged.
        :param client: The Bfabric client to use.
        """
        instance = cls(job=job, sample_dir=sample_dir, client=client)
        instance.stage_all()

    def stage_all(self) -> None:
        """Stages all required input files for a particular job."""
        self.stage_bfabric_inputs()
        self._standardize_input_table()
        self.stage_pipeline_parameters()

    def _standardize_input_table(self):
        input_path = self._sample_dir / "mass_list.unstandardized.raw.csv"
        output_path = self._sample_dir / "mass_list.raw.csv"
        copy_standardized_table(input_path, output_path)

    @property
    def _inputs_spec(self) -> dict[str, list[dict[str, str | int | bool]]]:
        return {
            "inputs": [
                {
                    "type": "bfabric_dataset",
                    "id": self._dataset_id,
                    "filename": "mass_list.unstandardized.raw.csv",
                    "separator": ",",
                },
                {
                    "type": "bfabric_resource",
                    "id": self._imzml_resource_id,
                    "filename": "raw.imzML",
                    "check_checksum": True,
                },
                {
                    "type": "bfabric_resource",
                    "id": self._ibd_resource_id,
                    "filename": "raw.ibd",
                    "check_checksum": True,
                },
            ]
        }

    def stage_bfabric_inputs(self) -> None:
        inputs_yaml = self._sample_dir / "inputs_spec.yml"
        with inputs_yaml.open("w") as file:
            yaml.safe_dump(self._inputs_spec, file)
        prepare_folder(
            inputs_yaml=inputs_yaml, target_folder=self._sample_dir, client=self._client, ssh_user=self._ssh_user
        )

    @cached_property
    def _ibd_resource_id(self) -> int:
        imzml_resource = Resource.find(id=self._imzml_resource_id, client=self._client)
        if imzml_resource["name"].endswith(".imzML"):
            expected_name = imzml_resource["name"][:-6] + ".ibd"
            results = self._client.read(
                "resource",
                {"name": expected_name, "containerid": imzml_resource["container"]["id"]},
                max_results=1,
                return_id_only=True,
            )
            return results[0]["id"]
        else:
            # TODO this will have to be refactored later
            raise NotImplementedError("Only .imzML files are supported for now")

    def stage_pipeline_parameters(self) -> None:
        """Copies the `pipeline_params.yml` file to the particular sample's directory."""
        output_path = self._sample_dir / "pipeline_params.yml"
        logger.debug(f"Staging pipeline parameters to {self._sample_dir}")
        with output_path.open("w") as file:
            yaml.dump(self._job.pipeline_parameters.model_dump(mode="json"), file)
