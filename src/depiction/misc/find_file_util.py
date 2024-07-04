from pathlib import Path
from typing import Optional


def find_one_by_glob(input_dir: str | Path, glob_pattern: str) -> Optional[str]:
    """
    Returns finds one file in input_dir matching the glob pattern.
    If there are multiple such files, an error is raised!
    :param input_dir: the directory to search in
    :param glob_pattern: the glob pattern to search for, e.g. "*.imzML"
    """
    candidates = list(Path(input_dir).glob(glob_pattern))
    if len(candidates) == 0:
        return None
    elif len(candidates) == 1:
        return str(candidates[0].absolute())
    else:
        raise RuntimeError(f"Multiple files with {glob_pattern=} found in {input_dir=}!")


def find_one_by_extension(input_dir: str | Path, extension: str) -> Optional[str]:
    """
    Returns finds one file in input_dir with the given extension.
    If there are multiple such files, an error is raised!
    :param input_dir: the directory to search in
    :param extension: the extension to search for, e.g. ".imzML"
    """
    return find_one_by_glob(input_dir=input_dir, glob_pattern=f"*{extension}")
