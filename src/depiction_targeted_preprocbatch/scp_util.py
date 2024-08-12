from __future__ import annotations

import subprocess
from pathlib import Path

from loguru import logger


def _is_remote(path: str | Path) -> bool:
    return ":" in str(path)


def scp(source: str | Path, target: str | Path, *, username: str | None = None) -> None:
    """Performs scp source target.
    Make sure that either the source or target specifies a host, otherwise you should just use shutil.copyfile.
    """
    source_remote = _is_remote(source)
    target_remote = _is_remote(target)
    if source_remote == target_remote:
        msg = f"Either source or target should be remote, but not both {source_remote=} == {target_remote=}"
        raise ValueError(msg)
    if username and source_remote:
        source = f"{username}@{source}"
    elif username and target_remote:
        target = f"{username}@{target}"

    logger.info(f"scp {source} {target}")
    subprocess.run(["scp", source, target], check=True)
