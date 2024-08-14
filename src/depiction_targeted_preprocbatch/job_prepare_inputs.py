from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from depiction.persistence.file_checksums import FileChecksums
from depiction_targeted_preproc.pipeline.setup import write_standardized_table
from depiction_targeted_preprocbatch.scp_util import scp
from loguru import logger

if TYPE_CHECKING:
    from depiction_targeted_preprocbatch.executor import BatchJob


class JobPrepareInputs:
    """Prepares the inputs for a particular job.
    :param job: The job to prepare the inputs for.
    :param sample_dir: The directory where the inputs should be staged.
    """

    def __init__(self, job: BatchJob, sample_dir: Path) -> None:
        self._job = job
        self._sample_dir = sample_dir

    @classmethod
    def prepare(cls, job: BatchJob, sample_dir: Path) -> None:
        """Prepares the inputs for a particular job.
        :param job: The job to prepare the inputs for.
        :param sample_dir: The directory where the inputs should be staged.
        """
        instance = cls(job=job, sample_dir=sample_dir)
        instance.stage_all()

    def stage_all(self) -> None:
        """Stages all required input files for a particular job."""
        self.stage_imzml()
        self.stage_panel()
        self.stage_pipeline_parameters()

    def stage_imzml(self) -> None:
        """Copies the `raw.imzML` and `raw.ibd` files to the sample directory.
        This method assumes the position will be on a remote server, and first needs to be copied with scp.
        """
        logger.debug(f"Staging imzML and ibd files for {self._job.imzml_relative_path}")

        # Check for some not-yet supported functionality (TODO)
        if self._job.imzml_relative_path.suffix != ".imzML":
            # TODO implement this later
            raise NotImplementedError(
                "Currently only .imzML files are supported, .imzML.zip will be supported in the future"
            )

        # determine the paths to copy from
        input_paths = [self._job.imzml_relative_path, self._job.imzml_relative_path.with_suffix(".ibd")]
        scp_uris = [f"{self._job.imzml_storage.scp_prefix}{path}" for path in input_paths]

        # perform the copies
        for scp_uri, result_name in zip(scp_uris, ["raw.imzML", "raw.ibd"]):
            scp(scp_uri, str(self._sample_dir / result_name), username=self._job.ssh_user)

        # check the checksum
        actual_checksum = FileChecksums(file_path=self._sample_dir / "raw.imzML").checksum_md5
        if actual_checksum != self._job.imzml_checksum:
            raise ValueError(f"Checksum mismatch: expected {self._job.imzml_checksum}, got {actual_checksum}")

    def stage_panel(self) -> None:
        """Writes the marker panel to the sample directory."""
        logger.debug(f"Staging panel for {self._job.imzml_relative_path}")
        write_standardized_table(input_df=self._job.panel_df, output_csv=self._sample_dir / "mass_list.raw.csv")

    def stage_pipeline_parameters(self) -> None:
        """Copies the `pipeline_params.yml` file to the particular sample's directory."""
        logger.debug(f"Staging pipeline parameters for {self._job.imzml_relative_path}")
        shutil.copyfile(
            self._sample_dir.parents[1] / "pipeline_params.yml",
            self._sample_dir / "pipeline_params.yml",
        )
