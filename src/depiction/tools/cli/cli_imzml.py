from loguru import logger
from pathlib import Path

import cyclopts

from depiction_io import get_read_file

cmd_imzml = cyclopts.App()


@cmd_imzml.command(name="verify")
def cmd_imzml_verify(imzml_path: Path) -> None | int:
    """Verifies if the .ibd file associated with the .imzML file has
    the correct checksum.
    :param imzml_path: Path to the .imzML file.
    """
    read_file = get_read_file(imzml_path)
    if read_file.is_checksum_valid:
        logger.success(f"Checksum for {imzml_path} is valid.")
    else:
        logger.error(f"Checksum for {imzml_path} is invalid.")
        return 1
