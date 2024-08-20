from __future__ import annotations

import zipfile
from pathlib import Path

from bfabric import Bfabric
from bfabric.entities import Storage, Resource
from loguru import logger

from depiction.persistence.file_checksums import FileChecksums
from depiction_targeted_preproc.app.workunit_config import WorkunitConfig
from depiction_targeted_preprocbatch.scp_util import scp


class JobExportResults:
    """Exports the results of the job to the output storage and registers it in B-Fabric."""

    def __init__(
        self,
        client: Bfabric,
        work_dir: Path,
        workunit_config: WorkunitConfig,
        output_storage: Storage,
        force_ssh_user: str | None = None,
    ) -> None:
        self._client = client
        self._workunit_config = workunit_config
        self.output_dir = work_dir / "output"
        self._output_storage = output_storage
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
            force_ssh_user=force_ssh_user,
        )
        instance.export_results(sample_name, result_files)

    def export_results(self, sample_name: str, result_files: list[Path]) -> None:
        """Exports the results of one job."""
        zip_file_path = self._create_zip_file(result_files, sample_name)
        output_path_relative = self._copy_zip_to_storage(zip_file_path)
        self._register_zip_in_workunit(output_path_relative, zip_file_path)

    def delete_local(self):
        # TODO this functionality will be needed when processing large jobs
        raise NotImplementedError

    def _create_zip_file(self, result_files: list[Path], sample_name: str) -> Path:
        """Creates a ZIP file containing the results for one sample, and returns the zip file's path."""
        self.output_dir.mkdir(exist_ok=True, parents=True)
        zip_file_path = self.output_dir / f"{sample_name}.zip"
        with zipfile.ZipFile(zip_file_path, "w") as zip_file:
            for result_file in result_files:
                zip_file.write(result_file, arcname=Path(sample_name) / result_file.name)
        return zip_file_path

    def _register_zip_in_workunit(self, output_path_relative: Path, zip_file_path: Path) -> None:
        checksum = FileChecksums(file_path=zip_file_path).checksum_md5
        # TODO this somehow seems to be executed multiple times and fails...
        self._client.save(
            "resource",
            {
                "name": zip_file_path.name,
                "workunitid": self._workunit_id,
                "storageid": self._output_storage.id,
                "relativepath": output_path_relative,
                "filechecksum": checksum,
                "status": "available",
                "size": zip_file_path.stat().st_size,
            },
        )

    def _copy_zip_to_storage(self, zip_file_path: Path) -> Path:
        output_path = self._workunit_config.output_folder_absolute_path / zip_file_path.name
        output_path_relative = output_path.relative_to(self._output_storage.base_path)
        output_uri = f"{self._output_storage.scp_prefix}{output_path_relative}"
        scp(zip_file_path, output_uri, username=self._force_ssh_user)
        return output_path_relative

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
