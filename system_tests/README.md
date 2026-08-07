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
`inputs/.gitignore` ignores everything in that directory except itself and `inputs.yml`, so
the three fixture files stay untracked.

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
hold for synthetic input. Both are real work and neither was done; the intended
route — a small, openly licensed acquisition downloaded and checksum-verified at test
time — is written up as Phase G in
[`docs/refactoring/ROADMAP.md`](../docs/refactoring/ROADMAP.md).

**Half of that already exists.** [`tests/real_data/`](../tests/real_data/) fetches two
redistributable acquisitions into `.test-data/` and reads them; one is MIT-licensed and
59 MB, which is the Phase G candidate. What is missing here is the other half: pointing
these tests at it, and rewriting the `128 x 137` / 118-channel / 10131-non-zero assertions
so they derive from the input. Import the manifest from `tests/real_data/datasets.py`
rather than copying the URLs.
