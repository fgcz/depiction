from __future__ import annotations

import subprocess
import zipfile
from pathlib import Path

from bfabric import Bfabric
from bfabric.entities import Storage
from loguru import logger

from depiction.persistence.file_checksums import FileChecksums
from depiction_targeted_preproc.app.workunit_config import WorkunitConfig


class JobExportResults:
    def __init__(self, client: Bfabric, work_dir: Path, workunit_config: WorkunitConfig) -> None:
        self._client = client
        self._workunit_config = workunit_config
        self.output_dir = work_dir / "output"

    def export(self, sample_name: str, result_files: list[Path], output_storage: Storage) -> None:
        """Exports the results of one job."""
        # TODO export result as a zip file for the particular sample
        # create the zip file
        self.output_dir.mkdir(exist_ok=True, parents=True)
        zip_file_path = self.output_dir / f"{sample_name}.zip"
        with zipfile.ZipFile(zip_file_path, "w") as zip_file:
            for result_file in result_files:
                zip_file.write(result_file, arcname=Path(sample_name) / result_file.name)

        # Copy the zip file
        output_path = self._workunit_config.output_folder_absolute_path / zip_file_path.name
        output_path_relative = output_path.relative_to(output_storage.base_path)
        output_uri = f"{output_storage.scp_prefix}{output_path_relative}"
        self._scp(zip_file_path, output_uri)

        # Register the zip file in the workunit
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

    def _scp(self, source: str | Path, target: str | Path) -> None:
        """Performs scp source target.
        Make sure that either the source or target specifies a host, otherwise you should just use shutil.copyfile.
        """
        # TODO this should be moved to a central location
        logger.info(f"scp {source} {target}")
        subprocess.run(["scp", source, target], check=True)
