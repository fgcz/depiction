# System tests

End-to-end runs of the `depiction_targeted_preproc` pipeline against a real
acquisition. Run them with:

```bash
nox -s system_tests
```

They are **not** part of the default `nox` sessions and are **not** run in CI — see
below.

## Required inputs

The tests need three files in `inputs/`, none of which are in the repository:

| File | Source |
|---|---|
| `tonsil.imzML` | B-Fabric resource `2445579` |
| `tonsil.ibd` | B-Fabric resource `2445566` |
| `panel.csv` | B-Fabric dataset `53798` |

The same list is in machine-readable form in [`inputs/inputs.yml`](inputs/inputs.yml).
Everything in `inputs/` except those two files is gitignored.

Without them the tests skip with a message naming what is missing.

## Why this is not in CI

`tonsil.ibd` alone is 1.26 GB, and the assertions in
`calibration/test_pipeline_calibration_only.py` are pinned to that specific
acquisition (`128 x 137` pixels, 118 channels, 10131 non-zero values). Neither the
file nor equivalent public data can be committed, and the synthetic corpus in
`tests/differential/` cannot substitute for it — it exercises the I/O layer, not the
calibration pipeline's numerical output.

Making these tests runnable in CI would mean either publishing a redistributable
acquisition of comparable size, or rewriting the test to assert on properties that
hold for synthetic input. Both are real work and neither was done; see
[`docs/refactoring/ROADMAP.md`](../docs/refactoring/ROADMAP.md).
