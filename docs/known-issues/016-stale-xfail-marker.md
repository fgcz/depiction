# A stale `xfail` marker hides a passing test

Severity: **low** | Status: open | Found: 2026-08-10
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

## Fix sketch

Remove the marker. If it was guarding an optional dependency (`ms-peak-picker` lives in its own
extra), replace it with an explicit skip condition on that import so the intent is legible:

```python
pytest.importorskip("ms_peak_picker")
```

Consider `xfail_strict = true` under `[tool.pytest.ini_options]` so an xpass fails the run
rather than being reported and ignored — that is the setting that stops this recurring.

## Notes

The 25 skips in the same run are legitimate: 21 are differential-corpus cases that only apply
to compressed or continuous inputs, and 2 are `alphapept`-gated
(see `022-alphapept-and-msi-hdf5-dead-coverage.md`). The remaining two are marked
`TODO fix later` (`test_filter_by_snr_threshold.py:84`) and `Reconsider`
(`test_image_normalization.py:64`).
