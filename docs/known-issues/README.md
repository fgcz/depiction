# Known issues

A pre-dormancy audit of the repository, 2026-08-10. One file per finding.

Every entry was reproduced or read in the source before being written down; each file states
what was actually run. Findings that could not be reproduced were dropped rather than recorded
with a hedge. This is a record of what is **known to be wrong**, not a backlog — several
entries are deliberately marked won't-fix, with the reasoning in the file.

Test baseline at the time of the audit: `721 passed, 25 skipped, 1 xpassed`, at `38d90ff`.
It is now **`736 passed, 25 skipped`** for `pytest tests` plus `90 passed, 1 skipped` for
`pytest pkgs/depiction_io/tests`: `+6` from #56, `+8` from the fixes below, and the `xpassed`
turned into a pass. Both need the real-data fixtures, or 38 of those passes become skips —
point `DEPICTION_TEST_DATA_DIR` at a directory that has them.

Line numbers are as of `38d90ff` on `dev`; where a change would move them, the entry names the
symbol as well. Entries closed since then say so in their `Status:` line and describe what was
actually done rather than what was proposed.

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

## Cheap — still open

| # | Issue | Where |
|---|---|---|
| [009](009-stale-build-dirs-resurrect-deleted-modules.md) | Stale `build/lib` can resurrect the deleted imzML parser | `build/`, `pkgs/*/build/` |
| [017](017-stale-branches-and-worktrees.md) | 19 stale branches and 2 stale worktrees | repo metadata |
| [018](018-evaluate-bins-always-computes-in-float32.md) | `evaluate_bins` float64 guard can never be true | `spectrum/evaluate_bins.py:41` |

009 and 017 are not code: one is `rm -rf` on untracked directories, the other deletes branches
and worktrees. Neither can live in a commit, which is why they outlived the batch that closed
the rest. The documentation half of 009 — a `pyproject.toml` comment that claimed more than it
delivered — is done.

018 is cheap to change but not cheap to *decide*: correcting the guard changes the numeric
output of every mean spectrum and every binned intensity, which is what `baseline-diff.md`
pinned. Read the file before touching it.

## Cheap — fixed

Closed as one batch. Each entry now records what was actually done, which in three cases
differs from what the entry originally proposed.

| # | Issue |
|---|---|
| [007](007-reference-peak-distances-out-of-bounds-read.md) | Out-of-bounds numba read fabricated calibration distances |
| [008](008-pipeline-config-uses-yaml-unsafe-load.md) | `yaml.unsafe_load` on a pipeline config |
| [010](010-create-imzml-pool-write-pool-always-raises.md) | `CreateImzmlPool.write_pool` always raised — ruff ate a local, twice |
| [011](011-imzy-dependency-has-no-upper-bound.md) | `imzy` dependency had no upper bound |
| [012](012-horizontal-concat-fails-on-unequal-heights.md) | `horizontal_concat` crashed on unequal heights |
| [013](013-imzml-zip-extract-returns-a-path-that-does-not-exist.md) | `ImzmlZip.extract` returned a path that did not exist |
| [014](014-depiction-io-declares-unused-cyclopts.md) | `depiction_io` declared `cyclopts` for a deleted entry point |
| [015](015-stale-documentation-claims.md) | Four documents described code that no longer exists |
| [016](016-stale-xfail-marker.md) | Stale `xfail` marker hid a passing test |
| [023](023-open-dependabot-alerts-in-the-lockfile.md) | Open Dependabot alerts pinned in the lockfile |

Three entries were wrong in their specifics, and the corrections are in the files:

- **010** proposed a fix that would not have worked, and the finding was incomplete. The
  `str()` in its sketch had no matching change to `pool_source_df`, whose `abs_path` column
  held `Path` objects — so the comparison would have matched nothing. And repairing the query
  only moved the crash one line: `_write_metadata` **segfaults** in pandas 2.3.3 on a `Path`
  column. `write_pool` had never run to completion, for two reasons rather than one.
- **012** named the wrong caller. `multi_channel_image_concatenation.py:88` calls
  `horizontal_concat` too.
- **016** proposed `pytest.importorskip("ms_peak_picker")`, which would have turned a passing
  test into a skipped one — the import is deferred, so the test does not need the package.

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

Filed before this audit, not duplicated here:

- ~~**#50**~~ — `CALIB_HEATMAP` failed on any non-square acquisition. **Closed by #56**, which
  replaced the plot rather than repairing it and removed the `CALIB_HEATMAP` artifact
  entirely. That shifted the line numbers in 004 and 020; both entries are anchored on symbol
  names for this reason.
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

**007 was fixed without re-running it.** The recorded baseline therefore predates that change.
The argument for not re-running was that the fix only alters output where a fabricated
distance previously passed the `max_distance_mz` gate, which the entry describes as occasional
spurious observations rather than a biased fit — but that is an argument, not a measurement.
Anyone re-running the baseline should expect 007 to be a possible source of difference, and
both prerequisites still exist locally: the tree pinned at `ed222b3` and the two fixtures.

## Method

Six parallel audits (core correctness, `depiction_io`, pipeline, dead code, packaging/CI,
documentation), each followed by an adversarial verification pass whose instruction was to
refute rather than confirm. 46 claims survived that pass; they were then deduplicated and
re-checked by hand, which removed three refuted claims and corrected two others that were
directionally right but wrong in their specifics. Corrections are noted inline in the files
that carry them (006, 011, 022).

Fixing the batch above found three more errors of the same kind (010, 012, 016, listed under
*Cheap — fixed*), which is the honest lesson about the method: a claim that survives an
adversarial read can still be wrong in the way that only shows up when someone writes the
patch. Every entry still open should be read as a lead, not a specification.

Known blind spot: the sweep checked declared dependencies, version floors and CI configuration
but never queried the security advisory surface. 023 was found afterwards, by accident, from a
`git push` warning. `gh api repos/fgcz/depiction/dependabot/alerts` was re-run on 2026-08-11
and reported one open alert (setuptools), now cleared. Run it again before archiving.
