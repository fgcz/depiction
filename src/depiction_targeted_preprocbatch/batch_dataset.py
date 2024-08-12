from __future__ import annotations

import polars as pl
from bfabric.entities import Dataset, Resource
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bfabric import Bfabric


@dataclass
class BatchDatasetJob:
    """Information of a single job in the batch dataset, but not yet including the full information e.g. parameters
    at workunit level.
    """

    imzml: Resource
    panel: Dataset


class BatchDataset:
    """Handles the batch dataset parsing."""

    def __init__(self, dataset_id: int, client: Bfabric) -> None:
        self.dataset_id = dataset_id
        self.client = client

    @cached_property
    def _dataset_df(self) -> pl.DataFrame:
        """Underlying dataset as a polars DataFrame."""
        return Dataset.find(id=self.dataset_id, client=self.client).to_polars()

    @cached_property
    def jobs(self) -> list[BatchDatasetJob]:
        """List of jobs in the batch dataset."""
        imzml_resources = Resource.find_all(ids=self._dataset_df["Imzml"].unique().to_list(), client=self.client)
        panel_datasets = Dataset.find_all(ids=self._dataset_df["PanelDataset"].unique().to_list(), client=self.client)
        return [
            BatchDatasetJob(
                imzml=imzml_resources[int(imzml_id)],
                panel=panel_datasets[int(panel_id)],
            )
            for imzml_id, panel_id in self._dataset_df[["Imzml", "PanelDataset"]].iter_rows()
        ]
