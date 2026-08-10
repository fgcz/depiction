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

## Why it is filed as won't-fix

The only in-repo callers are the sandbox and the `DEBUG` artifact. Repairing the three call
sites would hand a successor a *working* path to `maxmin_sampling`'s wrong geometry, which is
worse than an obvious `TypeError`. Fixing it properly means reviewing the sampling algorithm,
which is a research question, not a maintenance task.

## What to do instead

One line in `README.md` and/or `docs/refactoring/ROADMAP.md`:

> The clustering tools (`depiction.tools.clustering`, `depiction.clustering`) and
> `depiction_cluster_sandbox` are broken and unsupported. They fail immediately if called.

Then drop `cluster_default_hdbscan.png` from `PipelineArtifact.DEBUG` so the pipeline stops
routing users into it — see `004-debug-artifact-requires-dev-only-hdbscan.md`.

Deleting the whole surface is also defensible and would shrink the archive, but it is a larger
decision than a pre-dormancy sweep should make unilaterally.
