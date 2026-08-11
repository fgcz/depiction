# The clustering surface is dead, loudly

Severity: **medium** | Status: **won't fix — documented** | Found: 2026-08-10
Files: `src/depiction/tools/clustering.py:157,171`,
`src/depiction/clustering/maxmin_sampling.py:56`, `src/depiction_cluster_sandbox/`

## Symptom

Nothing in the clustering surface can succeed. It fails immediately and visibly, so nobody has
been silently misled — but a reader will waste time assuming it works.

- `depiction.tools.clustering` constructs `MultiChannelImage(...)` without the required
  `is_foreground` argument at both `:157` and `:171`, so every path raises
  `TypeError: MultiChannelImage.__init__() missing 1 required positional argument`.
- `clustering/maxmin_sampling.py:56` ignores its `metric` argument after the first pick and can
  return duplicate indices.
- `depiction_cluster_sandbox` needs a gitignored `data-sandbox/` directory that no longer
  exists, ships no `Snakefile` in the wheel (`[tool.setuptools.package-data]` covers
  `depiction_targeted_preproc` only), and imports `umap-learn`, which is declared in no extra.

## Correction, 2026-08-11: the pipeline's own two cluster scripts were not part of this

This entry originally implied that everything reachable from `PipelineArtifact.DEBUG` shared the
same fate. It does not. `workflow/proc/cluster_kmeans.py` and `workflow/proc/cluster_hdbscan.py`
had the *same* defect — each ended in a one-argument `MultiChannelImage(...)` — but they call
sklearn and `hdbscan` directly and never touch `depiction.tools.clustering` or
`clustering/maxmin_sampling.py`. Repairing them therefore exposes none of the doubtful geometry
this file is about, so both were fixed to use `MultiChannelImage.from_flat`, and
`tests/unit/targeted_preproc/workflow/proc/test_cluster.py` now covers them through to the PNG.

The audit missed that `cluster_kmeans.py` was broken at all — it recorded only the `hdbscan`
import problem, so it read as though the kmeans half of `DEBUG` worked.

`cluster_default_hdbscan.png` was dropped from `DEBUG` all the same, for the dependency reason in
the (now deleted) 004: `hdbscan` is declared only in the `dev` extra and publishes no manylinux
wheel, so promoting it would make every clean Linux install compile it. `proc_cluster_hdbscan` and
its script stay in-tree, repaired and runnable under `[dev]`; they are simply unreachable from any
artifact. Unlike the five rules in `020-dead-snakemake-rules.md`, that rule does run.

## Why the rest is filed as won't-fix

The remaining scope is `depiction.tools.clustering:157,171`, `clustering/maxmin_sampling.py:56`
and `depiction_cluster_sandbox`, whose only in-repo caller is the sandbox. Repairing those call
sites would hand a successor a *working* path to `maxmin_sampling`'s wrong geometry, which is
worse than an obvious `TypeError`. Fixing it properly means reviewing the sampling algorithm,
which is a research question, not a maintenance task.

## What to do instead

One line in `README.md` and/or `docs/refactoring/ROADMAP.md`:

> The clustering *tools* (`depiction.tools.clustering`, `depiction.clustering`) and
> `depiction_cluster_sandbox` are broken and unsupported. They fail immediately if called. The
> pipeline's own `proc/cluster_kmeans.py` and `proc/cluster_hdbscan.py` are separate and work.

Deleting the whole surface is also defensible and would shrink the archive, but it is a larger
decision than a pre-dormancy sweep should make unilaterally.
