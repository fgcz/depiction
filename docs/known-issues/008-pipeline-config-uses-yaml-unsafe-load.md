# Pipeline config is parsed with `yaml.unsafe_load`

Severity: **low** | Status: open | Found: 2026-08-10
File: `src/depiction_targeted_preproc/pipeline_config/model.py:20`

## Symptom

```python
return cls.model_validate(yaml.unsafe_load(path.read_text()))
```

`unsafe_load` constructs arbitrary Python objects, so a config file containing
`!!python/object/apply:os.system ["..."]` executes on load.

## Why it matters (and why it is only `low`)

The threat model is weak: the file lives in a pipeline working directory that the person
running the pipeline already controls, and anyone who can write it can usually run code anyway.
It is listed here because it is a **one-word fix with no downside**, and because it is the only
unsafe load left in the repo — every sibling call site already uses `safe_load`:

```
$ grep -rn "yaml.unsafe_load\|yaml.safe_load" src/depiction_targeted_preproc --include=*.py
pipeline_config/model.py:20:                    yaml.unsafe_load(...)   <- the only one
panel/standardize_input_panel.py:19:            yaml.safe_load(...)
pipeline_config/validate.py:14,21:              yaml.safe_load(...)
workflow/prepare_pipeline/write_pipeline_params.py:18:  yaml.safe_load(...)
workflow/vis/test_mass_shifts.py:22:            yaml.safe_load(...)
workflow/exp/prepare_calibration_config_with_no_smoothing.py:32: yaml.safe_load(...)
app_interface/process_chunk.py:20:              yaml.safe_load(...)
```

The asymmetry suggests `unsafe_load` was a leftover from debugging, not a decision.

## Fix sketch

Change `unsafe_load` to `safe_load`, then run the four shipped presets under
`pipeline_config/config_presets/` plus the system tests. All presets are plain YAML and
round-trip through `safe_load` unchanged.
