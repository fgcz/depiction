from __future__ import annotations

from pathlib import Path
from typing import Any

from loguru import logger
from pydantic import BaseModel

from bfabric import Bfabric
from bfabric.entities import Resource, Dataset
from bfabric.experimental.app_interface.workunit.definition import WorkunitDefinition


class DispatchIndividualResourcesConfig(BaseModel):
    input_resources_suffix_filter: str | None = ".imzML"
    dataset_resource_column: str = "Imzml"
    dataset_param_columns: list[tuple[str, str]] = [("PanelDataset", "mass_list_id")]


class DispatchIndividualResources:
    """Dispatches jobs on individual resources specified in the workunit."""

    def __init__(self, client: Bfabric, config: DispatchIndividualResourcesConfig, out_dir: Path) -> None:
        self._client = client
        self._config = config
        self._out_dir = out_dir

    def dispatch_job(self, resource: Resource, params: dict[str, Any]):
        raise NotImplementedError

    def dispatch_workunit(self, definition: WorkunitDefinition):
        params = definition.execution.raw_parameters
        if definition.execution.resources:
            self._dispatch_jobs_resource_flow(definition, params)
        elif definition.execution.dataset:
            self._dispatch_jobs_dataset_flow(definition, params)
        else:
            raise ValueError("either dataset or resources must be provided")

    def _dispatch_jobs_resource_flow(self, definition: WorkunitDefinition, params: dict[str, Any]):
        resources = Resource.find_all(ids=definition.execution.resources, client=self._client)
        for resource in sorted(resources.values()):
            if self._config.input_resources_suffix_filter is not None and not resource["name"].endswith(
                self._config.input_resources_suffix_filter
            ):
                logger.info(f"Skipping resource {resource['name']!r} as it does not match the extension filter.")
                continue
            self.dispatch_job(resource=resource, params=params)

    def _dispatch_jobs_dataset_flow(self, definition: WorkunitDefinition, params: dict[str, Any]):
        dataset = Dataset.find(id=definition.execution.dataset, client=self._client)
        dataset_df = dataset.to_polars()
        resources = Resource.find_all(
            ids=dataset_df[self._config.dataset_resource_column].unique().to_list(), client=self._client
        )
        for row in dataset_df.iter_rows(named=True):
            resource_id = row[self._config.dataset_resource_column]
            row_params = {name: row[dataset_name] for dataset_name, name in self._config.dataset_param_columns}
            self.dispatch_job(resource=resources[resource_id], params=params | row_params)
