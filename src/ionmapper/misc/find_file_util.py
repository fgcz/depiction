import os
import glob
from pathlib import Path
from typing import Optional
import shutil


def copy_imzml_file(source_imzml: Path, target_imzml: str) -> None:
    """
    Copies the imzML file and the corresponding ibd file to the target file.
    :param source_imzml: the source imzML file (including the .imzML extension)
    :param target_imzml: the target imzML file (including the .imzML extension)
    """
    shutil.copyfile(source_imzml, target_imzml)
    shutil.copyfile(
        source_imzml.with_suffix(".ibd"),
        target_imzml.replace(".imzML", ".ibd"),
    )


def find_one_by_glob(input_dir: str, glob_pattern: str) -> Optional[str]:
    """
    Returns finds one file in input_dir matching the glob pattern.
    If there are multiple such files, an error is raised!
    :param input_dir: the directory to search in
    :param glob_pattern: the glob pattern to search for, e.g. "*.imzML"
    """
    candidates = glob.glob(os.path.join(input_dir, glob_pattern))
    if len(candidates) == 0:
        return None
    elif len(candidates) == 1:
        return candidates[0]
    else:
        raise RuntimeError(f"Multiple files with {glob_pattern=} found in {input_dir=}!")


def find_one_by_extension(input_dir: str, extension: str) -> Optional[str]:
    """
    Returns finds one file in input_dir with the given extension.
    If there are multiple such files, an error is raised!
    :param input_dir: the directory to search in
    :param extension: the extension to search for, e.g. ".imzML"
    """
    return find_one_by_glob(input_dir=input_dir, glob_pattern=f"*{extension}")
