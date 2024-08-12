from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from bfabric.entities import Storage
from loguru import logger

from depiction.persistence.file_checksums import FileChecksums
from depiction_targeted_preproc.pipeline.setup import write_standardized_table

if TYPE_CHECKING:
    from depiction_targeted_preprocbatch.executor import BatchJob


class JobPrepareInputs:
    """Prepares the inputs for a particular job."""

    def __init__(self, job: BatchJob, sample_dir: Path) -> None:
        self._job = job
        self._sample_dir = sample_dir

    def prepare(self) -> None:
        """Stages all required input files for a particular job."""
        self._stage_imzml(
            relative_path=self._job.imzml_relative_path,
            input_storage=self._job.imzml_storage,
            sample_dir=self._sample_dir,
            checksum=self._job.imzml_checksum,
        )
        self._stage_panel(
            sample_dir=self._sample_dir,
            panel_df=self._job.panel_df,
        )
        self._stage_pipeline_parameters(sample_dir=self._sample_dir)

    def _stage_imzml(self, relative_path: Path, input_storage: Storage, sample_dir: Path, checksum: str) -> None:
        """Copies the `raw.imzML` and `raw.ibd` files to the sample directory.
        This method assumes the position will be on a remote server, and first needs to be copied with scp.
        :param relative_path: Relative path of the imzML file (relative to storage roo-).
        :param input_storage: Storage of the imzML file.
        :param sample_dir: Directory to copy the files to.
        :param checksum: Expected checksum of the imzML file.
        """
        # Check for some not-yet supported functionality (TODO)
        if relative_path.suffix != ".imzML":
            # TODO implement this later
            raise NotImplementedError(
                "Currently only .imzML files are supported, .imzML.zip will be supported in the future"
            )

        # determine the paths to copy from
        input_paths = [relative_path, relative_path.with_suffix(".ibd")]
        scp_uris = [f"{input_storage.scp_prefix}{path}" for path in input_paths]

        # perform the copies
        for scp_uri, result_name in zip(scp_uris, ["raw.imzML", "raw.ibd"]):
            self._scp(scp_uri, str(sample_dir / result_name))

        # check the checksum
        actual_checksum = FileChecksums(file_path=sample_dir / "raw.imzML").checksum_md5
        if actual_checksum != checksum:
            raise ValueError(f"Checksum mismatch: expected {checksum}, got {actual_checksum}")

    def _stage_panel(self, sample_dir: Path, panel_df: pl.DataFrame) -> None:
        """Writes the marker panel to the sample directory."""
        write_standardized_table(input_df=panel_df, output_csv=sample_dir / "mass_list.raw.csv")

    def _stage_pipeline_parameters(self, sample_dir: Path) -> None:
        """Copies the `pipeline_params.yml` file to the particular sample's directory."""
        shutil.copyfile(
            sample_dir.parents[1] / "pipeline_params.yml",
            sample_dir / "pipeline_params.yml",
        )

    def _scp(self, source: str | Path, target: str | Path) -> None:
        """Performs scp source target.
        Make sure that either the source or target specifies a host, otherwise you should just use shutil.copyfile.
        """
        # TODO this should be moved to a central location
        logger.info(f"scp {source} {target}")
        subprocess.run(["scp", source, target], check=True)
