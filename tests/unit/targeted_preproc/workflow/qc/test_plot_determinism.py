"""The QC plots must be a function of their input and nothing else.

Two of them were not. Run `CALIB_QC` twice on the same acquisition, in the same tree, and
`plot_marker_presence.pdf` and `plot_peak_density_grouped.pdf` came out visibly different --
52506 and 446952 differing subpixels at 150 dpi -- while the parquet tables they are drawn from
were byte-identical. A third, `plot_peak_density_combined.pdf`, turned out to be unstable too;
it survived the original two-run comparison by chance and only showed up on the third run.

The cause in every case was an ordering decision taken by polars, which does not promise one:
`group_by` returns its rows in a different order on each call, and `Series.unique()` does not
preserve order. That order then reached the figure -- as the category order of an axis, as the
stacking order of a bar, as which colour a variant was assigned.

These tests repeat each ordering decision many times in one process, which is enough: the
nondeterminism is per call, not per interpreter. `N_REPEATS` is well above the point where the
old code failed -- the unfixed `sorted_label_order` produced a different answer on essentially
every call -- so this is not a flaky test waiting to happen, it is a test that would have caught
the bug on its first run.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from depiction_targeted_preproc.workflow.qc.plot_marker_presence import sorted_label_order
from depiction_targeted_preproc.workflow.qc.plot_peak_density import density_by_group

N_REPEATS = 25


def _marker_frame() -> pl.DataFrame:
    """A frame shaped like the one `plot_marker_presence` builds, with the ties that matter.

    Half the labels share a `fraction`, which is what the real data looks like at the tightest
    cutoff: most markers are detected in either all or none of the spectra.
    """
    labels = [f"peak_{i:04d}" for i in range(40)]
    return pl.DataFrame(
        {
            "label": labels,
            "variant": ["calibrated"] * 40,
            "detection_dist": [0.005] * 40,
            "fraction": [1.0] * 20 + [0.0] * 20,
        }
    )


def _shuffled(df: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Permutes the rows, which is what `group_by` does to this frame in production.

    The shuffle is the whole point of the test. Calling `sorted_label_order` repeatedly on one
    fixed frame proves nothing -- `sort` is a pure function of its input -- and the pre-fix code
    passes that version. What varied in production was the *input* order, because the frame comes
    straight out of a `group_by`. Feeding a permutation each time reproduces that exactly: the old
    one-key sort returns a different answer on all 25, the two-key sort on none.
    """
    return df[list(rng.permutation(df.height))]


def test_sorted_label_order_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    df = _marker_frame()
    orders = {tuple(sorted_label_order(_shuffled(df, rng), cutoff=0.005)) for _ in range(N_REPEATS)}
    assert len(orders) == 1


def test_sorted_label_order_breaks_ties_by_label() -> None:
    """The tie-break is alphabetical within a fraction, and fraction still dominates."""
    order = sorted_label_order(_marker_frame(), cutoff=0.005)
    assert order[:20] == sorted(order[:20])
    assert order[20:] == sorted(order[20:])
    # the fraction=1.0 block must come first, descending
    assert order[0] == "peak_0000"
    assert order[20] == "peak_0020"


def test_density_by_group_is_deterministic() -> None:
    rng = np.random.default_rng(0)
    n = 20_000
    df = pl.DataFrame(
        {
            "dist": rng.normal(0, 0.1, n),
            "variant": rng.choice(["calibrated", "baseline_adj"], n),
            "mass_group": rng.choice(["group_0", "group_1", "group_2"], n),
        }
    )
    frames = [density_by_group(_shuffled(df, rng)) for _ in range(5)]
    for other in frames[1:]:
        assert frames[0].equals(other)


def test_density_by_group_returns_one_curve_per_pair() -> None:
    rng = np.random.default_rng(1)
    n = 5_000
    df = pl.DataFrame(
        {
            "dist": rng.normal(0, 0.1, n),
            "variant": rng.choice(["calibrated", "baseline_adj"], n),
            "mass_group": rng.choice(["group_0", "group_1"], n),
        }
    )
    out = density_by_group(df)
    assert out.select(["variant", "mass_group"]).unique().height == 4
    # sorted by mass_group, then variant, then dist -- the order the chart relies on
    assert out.equals(out.sort(["mass_group", "variant", "dist"]))
