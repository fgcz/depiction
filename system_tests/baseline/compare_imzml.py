"""Compares two imzML/ibd pairs by their decoded contents, under whichever reader is installed.

Runs in *both* environments -- this tree's imzy backend and the pre-refactor tree's
hand-rolled parser -- which is the point. A single reader comparing a pyimzml-written file
against an imzy-written one cannot distinguish "the files agree" from "the reader makes the
same mistake on both", and Phase E recorded losing exactly that check when the second parser
was deleted. Two independent XML parsers agreeing is the check; the import shim below is all
it costs.

The files are compared by value, never by bytes: the UUID, the SHA-1 and the `.ibd` layout
differ between two correct writers by construction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cyclopts
import numpy as np

try:  # this tree
    from depiction_io import get_read_file
except ImportError:  # the pre-refactor baseline
    from depiction.persistence import ImzmlReadFile

    def get_read_file(path: Path) -> ImzmlReadFile:
        return ImzmlReadFile(path)


app = cyclopts.App()


def describe_array(name: str, left: np.ndarray, right: np.ndarray) -> list[str]:
    """Everything that can differ between two arrays, in one place.

    dtype is reported alongside the values rather than folded into an equality check,
    because an intensity array that came back float32 instead of float64 is a real finding
    even when every value it can represent is identical.
    """
    problems = []
    if left.dtype != right.dtype:
        problems.append(f"{name}: dtype baseline={left.dtype} current={right.dtype}")
    if left.shape != right.shape:
        problems.append(f"{name}: shape baseline={left.shape} current={right.shape}")
        return problems
    if not np.array_equal(left, right, equal_nan=np.issubdtype(left.dtype, np.floating)):
        differing = int(np.sum(left != right))
        largest = float(np.nanmax(np.abs(left.astype(np.float64) - right.astype(np.float64))))
        problems.append(f"{name}: {differing}/{left.size} values differ, max abs diff {largest:.6g}")
    return problems


@app.default()
def main(baseline_imzml: Path, current_imzml: Path, *, report_every: int = 2000) -> None:
    """Compares two imzML files spectrum by spectrum and exits non-zero on any difference."""
    left_file, right_file = get_read_file(baseline_imzml), get_read_file(current_imzml)
    problems: list[str] = []

    print(f"reader: {type(left_file).__name__}")
    if left_file.n_spectra != right_file.n_spectra:
        problems.append(f"n_spectra: baseline={left_file.n_spectra} current={right_file.n_spectra}")
    if left_file.imzml_mode != right_file.imzml_mode:
        problems.append(f"imzml_mode: baseline={left_file.imzml_mode} current={right_file.imzml_mode}")
    problems += describe_array("coordinates", np.asarray(left_file.coordinates), np.asarray(right_file.coordinates))

    if not problems:
        # Streamed rather than collected: the tonsil is 10131 spectra of 31100 bins, which is
        # 2.5 GB of m/z per tree if both sides are held at once.
        with left_file.reader() as left, right_file.reader() as right:
            for index in range(left_file.n_spectra):
                for name, get in (("mz", "get_spectrum_mz"), ("int", "get_spectrum_int")):
                    problems += describe_array(
                        f"spectrum {index} {name}", getattr(left, get)(index), getattr(right, get)(index)
                    )
                if index % report_every == 0:
                    print(f"  ... {index}/{left_file.n_spectra}", flush=True)

    print(f"\n{baseline_imzml}\n{current_imzml}")
    if problems:
        print(f"DIFFERENT -- {len(problems)} problem(s):")
        for problem in problems[:50]:
            print(f"  {problem}")
        if len(problems) > 50:
            print(f"  ... and {len(problems) - 50} more")
        sys.exit(1)
    print(f"IDENTICAL -- {left_file.n_spectra} spectra, coordinates and both arrays match exactly")


if __name__ == "__main__":
    app()
