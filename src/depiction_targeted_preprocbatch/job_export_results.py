from __future__ import annotations

import zipfile
from pathlib import Path

import yaml
from bfabric import Bfabric
from bfabric.entities import Storage, Resource
from depiction_targeted_preproc.app.workunit_config import WorkunitConfig
from loguru import logger


class JobExportResults:
    """Exports the results of the job to the output storage and registers it in B-Fabric."""

    def __init__(
        self,
        client: Bfabric,
        work_dir: Path,
        workunit_config: WorkunitConfig,
        output_storage: Storage,
        sample_name: str,
        force_ssh_user: str | None = None,
    ) -> None:
        self._client = client
        self._workunit_config = workunit_config
        self.output_dir = work_dir / "output"
        self._output_storage = output_storage
        self._sample_name = sample_name
        self._force_ssh_user = force_ssh_user

    @property
    def _workunit_id(self) -> int:
        return self._workunit_config.workunit_id

    @classmethod
    def export(
        cls,
        client: Bfabric,
        work_dir: Path,
        workunit_config: WorkunitConfig,
        sample_name: str,
        result_files: list[Path],
        output_storage: Storage,
        force_ssh_user: str | None,
    ) -> None:
        """Exports the results of one job."""
        instance = cls(
            client=client,
            work_dir=work_dir,
            workunit_config=workunit_config,
            output_storage=output_storage,
            sample_name=sample_name,
            force_ssh_user=force_ssh_user,
        )
        instance.export_results(result_files)

    def export_results(self, result_files: list[Path]) -> None:
        """Exports the results of one job."""
        self._create_zip_file(result_files)
        self._register_result()

    def delete_local(self):
        # TODO this functionality will be needed when processing large jobs
        raise NotImplementedError

    @property
    def _zip_file_path(self) -> Path:
        return self.output_dir / f"{self._sample_name}.zip"

    def _create_zip_file(self, result_files: list[Path]) -> None:
        """Creates a ZIP file containing the results for one sample."""
        self.output_dir.mkdir(exist_ok=True, parents=True)
        with zipfile.ZipFile(self._zip_file_path, "w") as zip_file:
            for result_file in result_files:
                zip_entry_path = result_file.relative_to(self.output_dir.parent)
                zip_file.write(result_file, arcname=zip_entry_path)

    @property
    def _outputs_spec(self) -> dict[str, list[dict[str, str | int]]]:
        return {
            "outputs": [
                {
                    "type": "bfabric_copy_resource",
                    "local_path": str(self._zip_file_path),
                    "remote_path": str(self._zip_file_path.relative_to(self.output_dir)),
                    "workunit_id": self._workunit_id,
                    "storage_id": self._output_storage.id,
                }
            ]
        }

    def _register_result(self) -> None:
        outputs_yaml = self.output_dir / f"{self._sample_name}_outputs_spec.yml"
        with outputs_yaml.open("w") as file:
            yaml.safe_dump(self._outputs_spec, file)

    @staticmethod
    def delete_default_resource(workunit_id: int, client: Bfabric) -> bool:
        """Deletes the default resource created by the wrapper creator if it exists. Returns true if the resource was
        successfully deleted.
        """
        logger.warning(
            "Currently, the wrapper creator has a limitation that makes it impossible to remove "
            "this resource. This will be addressed in the future."
        )
        if False:
            resources = Resource.find_by(
                {"name": "MSI_Targeted_PreprocBatch 0 - resource", "workunitid": workunit_id}, client=client
            )
            if len(resources) == 1:
                resource_id = list(resources.values())[0].id
                logger.info(f"Deleting default resource with ID {resource_id}")
                result = client.delete("resource", resource_id, check=False)
                return result.is_success
            elif len(resources) > 1:
                raise ValueError("There should never be more than one default resource.")
            else:
                return False
