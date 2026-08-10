# Known issues

A pre-dormancy audit of the repository, 2026-08-10. One file per finding.

Every entry was reproduced or read in the source before being written down; each file states
what was actually run. Findings that could not be reproduced were dropped rather than recorded
with a hedge. This is a record of what is **known to be wrong**, not a backlog — several
entries are deliberately marked won't-fix, with the reasoning in the file.

Test baseline at the time of the audit: `721 passed, 25 skipped, 1 xpassed`.
Line numbers are as of `38d90ff` on `dev`; where a pending change would move them, the entry
names the symbol as well.

## Fix first

These either produce silently wrong scientific output or block a clean install.

| # | Issue | Severity | Where |
|---|---|---|---|
| [001](001-calibration-models-indexed-by-spectrum-id.md) | Calibration models are assigned to the wrong pixels | **critical** | `calibration/apply/apply_models.py:63` |
| [002](002-peak-filters-emit-non-monotonic-mz.md) | Peak filters emit non-monotonic m/z | high | `peak_filtering/filter_n_highest_intensity.py:27` |
| [003](003-pixel-size-fabricated-on-export.md) | Pixel size dropped on write, fabricated as 1 µm on export | high | `depiction_io/imzml/metadata.py:7` |
| [004](004-debug-artifact-requires-dev-only-hdbscan.md) | `DEBUG` artifact needs `hdbscan`, a dev-only extra | medium | `pipeline_config/artifacts_mapping.py:36` |
| [005](005-readme-status-and-missing-run-instructions.md) | README claims active development; no doc says how to run the pipeline | medium | `README.md:5,22` |
| [006](006-ram-backend-does-not-implement-protocols.md) | `Ram*` classes do not implement the advertised protocols | medium | `depiction_io/ram/ram_reader.py:14` |

001 and 002 are the two that corrupt output silently. If only one thing gets fixed, fix 001.

## Cheap

Each is well under an hour, most under fifteen minutes.

| # | Issue | Where |
|---|---|---|
| [007](007-reference-peak-distances-out-of-bounds-read.md) | Out-of-bounds numba read fabricates calibration distances | `calibration/spectrum/reference_peak_distances.py:45` |
| [008](008-pipeline-config-uses-yaml-unsafe-load.md) | `yaml.unsafe_load` on a pipeline config | `pipeline_config/model.py:20` |
| [009](009-stale-build-dirs-resurrect-deleted-modules.md) | Stale `build/lib` can resurrect the deleted imzML parser | `build/`, `pkgs/*/build/` |
| [010](010-create-imzml-pool-write-pool-always-raises.md) | `CreateImzmlPool.write_pool` always raises — ruff ate a local, twice | `tools/create_imzml_pool.py:67` |
| [011](011-imzy-dependency-has-no-upper-bound.md) | `imzy` dependency has no upper bound | `pkgs/depiction_io/pyproject.toml:24` |
| [012](012-horizontal-concat-fails-on-unequal-heights.md) | `horizontal_concat` crashes on unequal heights | `image/horizontal_concat.py:27` |
| [013](013-imzml-zip-extract-returns-a-path-that-does-not-exist.md) | `ImzmlZip.extract` returns a path that does not exist | `depiction_io/imzml_zip.py:41` |
| [014](014-depiction-io-declares-unused-cyclopts.md) | `depiction_io` declares `cyclopts` for a deleted entry point | `pkgs/depiction_io/pyproject.toml:20` |
| [015](015-stale-documentation-claims.md) | Four documents describe code that no longer exists | various |
| [016](016-stale-xfail-marker.md) | Stale `xfail` marker hides a passing test | `tests/unit/tools/pick_peaks/test_pick_peaks.py` |
| [017](017-stale-branches-and-worktrees.md) | 19 stale branches and 2 stale worktrees | repo metadata |
| [018](018-evaluate-bins-always-computes-in-float32.md) | `evaluate_bins` float64 guard can never be true | `spectrum/evaluate_bins.py:41` |
| [023](023-open-dependabot-alerts-in-the-lockfile.md) | Two open Dependabot alerts pinned in the lockfile | `uv.lock` |

018 is cheap to change but not cheap to *decide* — read the file before touching it.

## Accepted — documented, not fixed

Real, verified, and deliberately left alone. Each file explains why fixing costs more than it
returns for an unmaintained repo.

| # | Issue |
|---|---|
| [019](019-clustering-surface-is-dead.md) | The clustering surface is dead, loudly |
| [020](020-dead-snakemake-rules.md) | A graveyard of dead Snakemake rules and workflow scripts |
| [021](021-ci-does-not-install-from-the-lockfile.md) | CI never installs from `uv.lock`; two default sessions need the network |
| [022](022-alphapept-and-msi-hdf5-dead-coverage.md) | `alphapept`- and `awkward`-gated code has no coverage |

## Tracked as GitHub issues instead

Filed before this audit, still open, not duplicated here:

- **#50** — `CALIB_HEATMAP` fails on any non-square acquisition. `plot_calibration_map.py:37`
  stacks the two axis coordinate vectors instead of building a meshgrid, so `np.stack` requires
  them to be equal length. Unexercised because `CALIB_HEATMAP` is not in the default
  `requested_artifacts`. **Fix in draft PR #56**, which replaces the plot rather than repairing
  it and removes the `CALIB_HEATMAP` artifact entirely. When that merges, the line numbers in
  004 and 020 shift — both entries are anchored on symbol names for this reason.
- **#51** — two QC plots are not reproducible run-to-run (`vl-convert` emits the PDF
  `/XObject` dictionary in a different order each time), which makes QC output unusable as a
  regression check. The underlying data is byte-identical.
- **#32** — high RAM usage in some pipeline steps.
- **#17** — documentation: installation, pipeline configuration, editing the workflow.

## Before changing numeric output

001, 002, 007 and 018 all change pipeline output on at least some inputs. The pre-refactor
baseline in [`../refactoring/baseline-diff.md`](../refactoring/baseline-diff.md) is the only
thing distinguishing an intended change from a regression — re-run it after any of them, and
record the new expected output.

## Method

Six parallel audits (core correctness, `depiction_io`, pipeline, dead code, packaging/CI,
documentation), each followed by an adversarial verification pass whose instruction was to
refute rather than confirm. 46 claims survived that pass; they were then deduplicated and
re-checked by hand, which removed three refuted claims and corrected two others that were
directionally right but wrong in their specifics. Corrections are noted inline in the files
that carry them (006, 011, 022).

Known blind spot: the sweep checked declared dependencies, version floors and CI configuration
but never queried the security advisory surface. 023 was found afterwards, by accident, from a
`git push` warning. Re-run `gh api repos/fgcz/depiction/dependabot/alerts` before archiving.
