from pathlib import Path

import polars as pl
import pytest
from depiction_targeted_preproc.app_interface.dispatch_individual_resources import (
    DispatchIndividualResources,
    config_msi_imzml,
)

from bfabric import Bfabric
from bfabric.entities import Resource, Dataset


@pytest.fixture()
def mock_client(mocker):
    return mocker.MagicMock(name="mock_client", spec=Bfabric)


@pytest.fixture()
def mock_config():
    return config_msi_imzml()


@pytest.fixture()
def mock_dispatch(mock_client, mock_config) -> DispatchIndividualResources:
    return DispatchIndividualResources(client=mock_client, config=mock_config, out_dir=Path("/dev/null"))


def test_dispatch_workunit_when_resources(mocker, mock_dispatch):
    mock_definition = mocker.MagicMock(name="mock_definition")
    mock_definition.execution.resources = [1, 2, 3]
    dispatch_jobs = mocker.patch.object(mock_dispatch, "_dispatch_jobs_resource_flow")
    write_chunks = mocker.patch.object(mock_dispatch, "_write_chunks")
    write_workunit_definition = mocker.patch.object(mock_dispatch, "_write_workunit_definition")
    mock_dispatch.dispatch_workunit(definition=mock_definition)
    dispatch_jobs.assert_called_once_with(mock_definition, mock_definition.execution.raw_parameters)
    write_chunks.assert_called_once_with(chunks=mock_dispatch._dispatch_jobs_resource_flow.return_value)
    write_workunit_definition.assert_called_once_with(definition=mock_definition)


def test_dispatch_workunit_when_dataset(mocker, mock_dispatch):
    mock_definition = mocker.MagicMock(name="mock_definition")
    mock_definition.execution.resources = []
    mock_definition.execution.dataset = 1
    dispatch_jobs = mocker.patch.object(mock_dispatch, "_dispatch_jobs_dataset_flow")
    write_chunks = mocker.patch.object(mock_dispatch, "_write_chunks")
    write_workunit_definition = mocker.patch.object(mock_dispatch, "_write_workunit_definition")
    mock_dispatch.dispatch_workunit(definition=mock_definition)
    dispatch_jobs.assert_called_once_with(mock_definition, mock_definition.execution.raw_parameters)
    write_chunks.assert_called_once_with(chunks=mock_dispatch._dispatch_jobs_dataset_flow.return_value)
    write_workunit_definition.assert_called_once_with(definition=mock_definition)


def test_dispatch_workunit_invalid_input(mocker, mock_dispatch):
    mock_definition = mocker.MagicMock(name="mock_definition")
    mock_definition.execution.resources = []
    mock_definition.execution.dataset = None
    with pytest.raises(ValueError, match="either dataset or resources must be provided"):
        mock_dispatch.dispatch_workunit(definition=mock_definition)


def test_dispatch_jobs_resource_flow(mocker, mock_dispatch, mock_client):
    mock_definition = mocker.MagicMock(name="mock_definition")
    mock_definition.execution.resources = [1, 2, 3]
    mock_definition.execution.raw_parameters = {"param1": "value1"}

    mock_resources = {
        1: Resource({"id": 1, "name": "resource1.imzML"}),
        2: Resource({"id": 2, "name": "resource2.imzML"}),
        3: Resource({"id": 3, "name": "resource3.txt"}),
    }
    mocker.patch.object(Resource, "find_all", return_value=mock_resources)

    mock_dispatch_job = mocker.patch.object(mock_dispatch, "dispatch_job")

    mock_dispatch._dispatch_jobs_resource_flow(mock_definition, mock_definition.execution.raw_parameters)

    assert mock_dispatch_job.call_count == 2
    mock_dispatch_job.assert_any_call(resource=mock_resources[1], params={"param1": "value1"})
    mock_dispatch_job.assert_any_call(resource=mock_resources[2], params={"param1": "value1"})


def test_dispatch_jobs_dataset_flow(mocker, mock_dispatch, mock_client):
    mock_definition = mocker.MagicMock(name="mock_definition")
    mock_definition.execution.dataset = 1
    mock_definition.execution.raw_parameters = {"param1": "value1"}

    mock_dataset = mocker.MagicMock(spec=Dataset)
    mock_dataset.to_polars.return_value = pl.DataFrame({"Imzml": [1, 2], "PanelDataset": ["panel1", "panel2"]})
    mocker.patch.object(Dataset, "find", return_value=mock_dataset)

    mock_resources = {
        1: Resource({"id": 1, "name": "resource1.imzML"}),
        2: Resource({"id": 2, "name": "resource2.imzML"}),
    }
    mocker.patch.object(Resource, "find_all", return_value=mock_resources)

    mock_dispatch_job = mocker.patch.object(mock_dispatch, "dispatch_job")

    mock_dispatch._dispatch_jobs_dataset_flow(mock_definition, mock_definition.execution.raw_parameters)

    assert mock_dispatch_job.call_count == 2
    mock_dispatch_job.assert_any_call(resource=mock_resources[1], params={"param1": "value1", "mass_list_id": "panel1"})
    mock_dispatch_job.assert_any_call(resource=mock_resources[2], params={"param1": "value1", "mass_list_id": "panel2"})


def test_dispatch_job(mock_dispatch):
    with pytest.raises(NotImplementedError):
        mock_dispatch.dispatch_job(resource=None, params={})
