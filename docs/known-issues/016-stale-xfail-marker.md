# A stale `xfail` marker hides a passing test

Severity: **low** | Status: **fixed** (#59) | Found: 2026-08-10
File: `tests/unit/tools/pick_peaks/test_pick_peaks.py::test_get_peak_picker_when_ms_peak_picker`

## Symptom

The suite reports one `xpassed`:

```
$ .venv/bin/python -m pytest tests -q -p no:randomly -p no:pretty -rxXs
...
XPASS tests/unit/tools/pick_peaks/test_pick_peaks.py::test_get_peak_picker_when_ms_peak_picker
721 passed, 25 skipped, 1 xpassed
```

The test now passes, but the `xfail` marker means a future *real* regression in that path
would be reported as an expected failure and stay invisible.

## Fix

The marker is gone and `xfail_strict = true` is set under `[tool.pytest.ini_options]`, so an
xpass now fails the run rather than being reported and ignored. That is the setting that stops
this recurring; it was the only `xfail` in the repo, so nothing else was affected.

The `pytest.importorskip("ms_peak_picker")` this file originally suggested would have been
**wrong**: `ms_peak_picker` is not installed in a default `--extra testing` environment, yet
the test passes there, because `MSPeakPicker` defers that import into `pick_peaks`
(`spectrum/peak_picking/ms_peak_picker_wrapper.py:28`) and the test only constructs one. An
`importorskip` would have turned a passing test into a skipped one.

## Notes

The 25 skips in the same run are legitimate: 21 are differential-corpus cases that only apply
to compressed or continuous inputs, and 2 are `alphapept`-gated
(see `022-alphapept-and-msi-hdf5-dead-coverage.md`). The remaining two are marked
`TODO fix later` (`test_filter_by_snr_threshold.py:84`) and `Reconsider`
(`test_image_normalization.py:64`).
