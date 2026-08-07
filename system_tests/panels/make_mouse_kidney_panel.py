"""Regenerates `mouse_kidney.csv`, the panel the public system-test fixture runs against.

**This is not a biological panel.** The masses below are simply the strongest well-separated
peaks in the fixture's own mean spectrum. They identify nothing; they exist so that the
targeted pipeline has reference masses that actually hit signal, which is what makes the
run exercise the code rather than propagate NaNs.

That last part is load-bearing. `CalibrationMethodConstantGlobalShift.preprocess_image_features`
takes `np.nanmedian` over the per-reference peak distances, so a panel whose masses have no
peak inside the +/- 2 Da search window yields a NaN shift, NaN m/z arrays, and a failure a
long way from its cause. Picking the peaks out of the data is the cheapest way to not have
that problem.

The FGCZ tonsil panel cannot be used here: it is B-Fabric dataset 53798, is gitignored along
with the rest of `system_tests/inputs/`, and is not ours to publish.

The output is committed -- it is under a kilobyte, and a pinned input is worth more to a test
than a regenerated one. This script exists so that the pin is reproducible rather than
mysterious. Run it with the fixture downloaded (see `tests/real_data/fetch.py`):

    uv run python -m system_tests.panels.make_mouse_kidney_panel
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from depiction_io import ImzmlModeEnum, get_read_file
from tests.real_data.datasets import MOUSE_KIDNEY

OUTPUT_PATH = Path(__file__).parent / "mouse_kidney.csv"

#: Spectra averaged to find the peaks. The fixture has 1581; a couple of hundred spread
#: across the acquisition is enough to make the strong peaks obvious and keeps this quick.
N_SPECTRA_SAMPLED = 200

#: Minimum spacing between selected masses, in Da. Wider than the calibration method's
#: +/- 2 Da search window, so that no two reference masses compete for the same peak.
MIN_SEPARATION_MZ = 5.0

N_MASSES = 20


def mean_spectrum(imzml_path: Path) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """The m/z axis and the mean intensity over a spread of spectra.

    Continuous mode only -- averaging presumes one shared m/z axis, and on a processed file
    this would silently sum arrays that do not line up.
    """
    read_file = get_read_file(imzml_path)
    if read_file.imzml_mode != ImzmlModeEnum.CONTINUOUS:
        raise ValueError(f"{imzml_path} is {read_file.imzml_mode}, and averaging needs a shared m/z axis.")
    indices = np.linspace(0, read_file.n_spectra - 1, N_SPECTRA_SAMPLED).astype(int)
    with read_file.reader() as reader:
        mz_arr = reader.get_spectrum_mz(int(indices[0]))
        total = np.zeros(len(mz_arr), dtype=np.float64)
        for index in indices:
            total += reader.get_spectrum_int(int(index))
    return np.asarray(mz_arr, dtype=np.float64), total / len(indices)


def select_masses(mz_arr: NDArray[np.float64], intensities: NDArray[np.float64]) -> NDArray[np.float64]:
    """The `N_MASSES` strongest local maxima, no two closer than `MIN_SEPARATION_MZ`."""
    is_local_max = np.r_[False, (intensities[1:-1] > intensities[:-2]) & (intensities[1:-1] >= intensities[2:]), False]
    candidates = np.flatnonzero(is_local_max)
    selected: list[float] = []
    for index in candidates[np.argsort(intensities[candidates])[::-1]]:
        mz = float(mz_arr[index])
        if all(abs(mz - chosen) >= MIN_SEPARATION_MZ for chosen in selected):
            selected.append(mz)
        if len(selected) == N_MASSES:
            break
    return np.sort(np.array(selected))


def label_for(mz: float) -> str:
    """A name that says where the mass came from, since it does not name a molecule."""
    return f"peak_{mz:.2f}".replace(".", "_")


def main() -> None:
    if not MOUSE_KIDNEY.is_available:
        raise SystemExit(
            f"{MOUSE_KIDNEY.name} is not downloaded. Run "
            f"`uv run python -m tests.real_data.fetch {MOUSE_KIDNEY.name}` first."
        )
    mz_arr, intensities = mean_spectrum(MOUSE_KIDNEY.imzml.local_path)
    masses = select_masses(mz_arr, intensities)
    with OUTPUT_PATH.open("w", newline="") as file:
        # csv's default dialect terminates lines with CRLF on every platform, which the
        # repository's line-ending hook would then rewrite behind this script's back.
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["label", "mass"])
        writer.writerows([label_for(mz), f"{mz:.4f}"] for mz in masses)
    print(f"Wrote {len(masses)} masses to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
