from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from bfabric.entities import Resource
from depiction_targeted_preproc.app.workunit_config import WorkunitConfig
from depiction_targeted_preproc.pipeline_config.model import PipelineParameters


@dataclass
class BatchJob:
    """Defines one job to be executed, including all required information."""

    imzml_resource_id: int
    pipeline_parameters: PipelineParameters
    dataset_id: int
    sample_name: str
    ssh_user: str | None = None

    @classmethod
    def from_bfabric(cls, imzml_resource: Resource, workunit_config: WorkunitConfig, ssh_user: str | None) -> BatchJob:
        return cls(
            imzml_resource_id=imzml_resource.id,
            pipeline_parameters=workunit_config.pipeline_parameters,
            dataset_id=workunit_config.input_dataset_id,
            sample_name=Path(imzml_resource["name"]).stem,
            ssh_user=ssh_user,
        )
