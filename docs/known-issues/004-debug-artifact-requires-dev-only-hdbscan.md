# `PipelineArtifact.DEBUG` requires `hdbscan`, which ships only in the `dev` extra

Severity: **medium** | Status: open | Found: 2026-08-10
Files: the `cluster_default_hdbscan.png` entry of `PipelineArtifact.DEBUG` in
`src/depiction_targeted_preproc/pipeline_config/artifacts_mapping.py` (line 36 on `38d90ff`,
line 31 once draft PR #56 removes the `CALIB_HEATMAP` block above it),
`src/depiction_targeted_preproc/workflow/proc/cluster_hdbscan.py:4`

## Symptom

On a clean install of `depiction` (no `[dev]`), requesting the `DEBUG` artifact makes the
pipeline die with `ModuleNotFoundError: No module named 'hdbscan'` — but only **after**
calibration and image generation have already run, so the compute is wasted.

## Why it happens

`artifacts_mapping.py` lists `cluster_default_hdbscan.png` under `PipelineArtifact.DEBUG`.
The rule producing it runs `workflow/proc/cluster_hdbscan.py`, which imports `hdbscan` at
module level. `hdbscan` is declared only in the `dev` extra:

```
$ python -c "import importlib.metadata as m; print([r for r in m.requires('depiction') if 'hdbscan' in r])"
['hdbscan; extra == "dev"']
```

This is the one dependency defect on the list reachable through the **supported public API**:
`params.yml` → `requested_artifacts: [DEBUG]` → `app_interface/process_chunk.py`.

## How to reproduce

Install without extras, then request `DEBUG` in a chunk's `params.yml` and run
`process_chunk`.

## Fix sketch

Drop `cluster_default_hdbscan.png` from the `DEBUG` list. Given that the whole clustering
surface is broken anyway (see `019-clustering-surface-is-dead.md`), removing the artifact is
more honest than promoting `hdbscan` to a runtime dependency.

If you prefer to keep it, move `hdbscan` into the base dependencies — but then it becomes a
hard install requirement for a code path that cannot currently succeed.

## Notes

Related but distinct: `holoviews`, `umap-learn`, `awkward` and `rpy2` are imported by shipped
modules and declared in **no** extra at all. Those are only reachable from dead rules and the
sandbox, so they are a documentation matter rather than a fix —
see `020-dead-snakemake-rules.md` and `022-alphapept-and-msi-hdf5-dead-coverage.md`.
