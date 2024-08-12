from __future__ import annotations

import zipfile
from pathlib import Path

from bfabric import Bfabric
from bfabric.entities import Storage

from depiction.persistence.file_checksums import FileChecksums
from depiction_targeted_preproc.app.workunit_config import WorkunitConfig
from depiction_targeted_preprocbatch.scp_util import scp


class JobExportResults:
    """Exports the results of the job to the output storage and registers it in B-Fabric."""

    def __init__(self, client: Bfabric, work_dir: Path, workunit_config: WorkunitConfig) -> None:
        self._client = client
        self._workunit_config = workunit_config
        self.output_dir = work_dir / "output"

    @classmethod
    def export(
        cls,
        client: Bfabric,
        work_dir: Path,
        workunit_config: WorkunitConfig,
        sample_name: str,
        result_files: list[Path],
        output_storage: Storage,
    ) -> None:
        """Exports the results of one job."""
        instance = cls(client=client, work_dir=work_dir, workunit_config=workunit_config)
        instance.export_results(sample_name, result_files, output_storage)

    def export_results(self, sample_name: str, result_files: list[Path], output_storage: Storage) -> None:
        """Exports the results of one job."""
        zip_file_path = self._create_zip_file(result_files, sample_name)
        output_path_relative = self._copy_zip_to_storage(zip_file_path, output_storage)
        self._register_zip_in_workunit(output_path_relative, output_storage, zip_file_path)

    def _create_zip_file(self, result_files, sample_name):
        self.output_dir.mkdir(exist_ok=True, parents=True)
        zip_file_path = self.output_dir / f"{sample_name}.zip"
        with zipfile.ZipFile(zip_file_path, "w") as zip_file:
            for result_file in result_files:
                zip_file.write(result_file, arcname=Path(sample_name) / result_file.name)
        return zip_file_path

    def _register_zip_in_workunit(self, output_path_relative, output_storage, zip_file_path):
        checksum = FileChecksums(file_path=zip_file_path).checksum_md5
        self._client.save(
            "resource",
            {
                "name": zip_file_path.name,
                "workunitid": self._workunit_config.workunit_id,
                "storageid": output_storage.id,
                "relativepath": output_path_relative,
                "filechecksum": checksum,
                "status": "available",
                "size": zip_file_path.stat().st_size,
            },
        )

    def _copy_zip_to_storage(self, zip_file_path, output_storage):
        output_path = self._workunit_config.output_folder_absolute_path / zip_file_path.name
        output_path_relative = output_path.relative_to(output_storage.base_path)
        output_uri = f"{output_storage.scp_prefix}{output_path_relative}"
        scp(zip_file_path, output_uri)
        return output_path_relative
