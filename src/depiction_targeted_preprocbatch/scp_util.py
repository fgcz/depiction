from __future__ import annotations

import subprocess
from pathlib import Path

from loguru import logger


def scp(source: str | Path, target: str | Path) -> None:
    """Performs scp source target.
    Make sure that either the source or target specifies a host, otherwise you should just use shutil.copyfile.
    """
    logger.info(f"scp {source} {target}")
    subprocess.run(["scp", source, target], check=True)
