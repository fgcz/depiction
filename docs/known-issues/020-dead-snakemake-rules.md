# A graveyard of dead Snakemake rules and workflow scripts

Severity: **medium** | Status: **won't fix — documented** | Found: 2026-08-10
Files: `src/depiction_targeted_preproc/workflow/experimental.smk:114`,
`workflow/proc/cluster_stats.py:10`, `workflow/vis/images_ome_ngff.py:2`,
`workflow/rules/rules_qc.smk` (rule `qc_plot_sample_spectra_before_after`),
`workflow/rules/rules_simulate.smk`

## Symptom

Five workflow components cannot run. All fail loudly at import or at DAG construction, and all
are unreachable from any `PipelineArtifact` — the corresponding lines in
`pipeline_config/artifacts_mapping.py` are commented out. Verified by importing each module:

```
$ .venv/bin/python -c "import depiction_targeted_preproc.workflow.proc.cluster_stats"
ModuleNotFoundError: No module named
  'depiction_targeted_preproc.workflow.proc.__cluster_stats'

$ .venv/bin/python -c "import depiction_targeted_preproc.workflow.vis.images_ome_ngff"
ImportError: cannot import name 'OmeZarrWriter' from 'bioio.writers'
```

- **`proc/cluster_stats.py:10`** imports `__cluster_stats`, a module that was never committed.
  The rule `proc_cluster_stats` is live in the workflow, so it has never been runnable.
- **`vis/images_ome_ngff.py:2`** imports `OmeZarrWriter`, renamed by an upstream `bioio` API
  change. `artifacts_mapping.py:19` has `images_default.ome.zarr` commented out of
  `CALIB_IMAGES`, so the default pipeline never requests it.
- **`experimental.smk:114`** cannot be parsed at all — duplicate rule name
  `vis_test_mass_shifts`.
- **`rules_qc.smk`, rule `qc_plot_sample_spectra_before_after`** (line 117 on `38d90ff`, 105
  once draft PR #56 lands) requires a `{sample}/peaks.imzML` that no rule produces.
- **`rules_simulate.smk`** wants a `{sample}_sim/full.csv` that nothing writes.

## Why it is filed as won't-fix

Five separate repairs, none of which any supported artifact needs, all of which fail
immediately rather than producing wrong output. The value is in telling a reader they are dead,
not in resurrecting them.

## What to do instead

Add a short "known-dead rules — do not use" block to
`src/depiction_targeted_preproc/README.md` (which currently reads `TODO: Add a description` in
full — see `015-stale-documentation-claims.md`).

If deleting feels better than documenting, the three cheapest to remove outright are
`experimental.smk`, `proc/cluster_stats.py` and `vis/images_ome_ngff.py`; nothing reachable
depends on them.

## Notes

`workflow/qc/plot_scan_direction.py:1` carries
`# TODO has been migrated/refactored in the new workflow folder`, suggesting more of this
directory is superseded than the five above. Not audited exhaustively.
