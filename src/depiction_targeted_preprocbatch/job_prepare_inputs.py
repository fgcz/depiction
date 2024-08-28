from __future__ import annotations

import shutil
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from bfabric import Bfabric
from bfabric.entities import Resource
from bfabric.experimental.app_interface.input_preparation.prepare import PrepareInputs
from bfabric.experimental.app_interface.input_preparation.specs import Specs
from depiction_targeted_preproc.pipeline.setup import copy_standardized_table
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

    def stage_bfabric_inputs(self) -> None:
        specs = Specs.model_validate(
            {
                "specs": [
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
        ).specs
        PrepareInputs(client=self._client, working_dir=self._sample_dir, ssh_user=self._ssh_user).prepare_all(
            specs=specs
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
                return_only_ids=True,
            )
            return results[0]["id"]
        else:
            # TODO this will have to be refactored later
            raise NotImplementedError("Only .imzML files are supported for now")

    def stage_pipeline_parameters(self) -> None:
        """Copies the `pipeline_params.yml` file to the particular sample's directory."""
        logger.debug(f"Staging pipeline parameters for {self._job.imzml_relative_path}")
        shutil.copyfile(
            self._sample_dir.parents[1] / "pipeline_params.yml",
            self._sample_dir / "pipeline_params.yml",
        )
