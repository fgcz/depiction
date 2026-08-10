# Stale branches and registered worktrees

Severity: **low** | Status: open | Found: 2026-08-10
Scope: repository metadata, no code

## Symptom

Twelve local and seven remote branches remain after their pull requests were squash-merged.
Because squash-merging rewrites history, `git branch --merged dev` reports **none** of them as
merged, so the repo looks like it has a dozen unfinished lines of work. For a dormant repo
that is actively misleading — the first thing a successor does is inspect those branches.

Every one maps to a merged PR:

| branch | PR |
|---|---|
| `test/imzml-write-file-integration` | #55 |
| `chore/retire-bfabric-app-interface` | #54 |
| `spatial-tools` | #53 |
| `workflow-cores` | #52 |
| `consumer-guidance` | #49 |
| `dependency-hygiene` | #48 |
| `vendor-snakemake-invoke` | #46 |
| `baseline-diff` | #45 |
| `public-fixture-system-tests` | #43 |
| `real-data-reader-checks` | #41 |
| `imzy-default` | #40 |
| `imzy-backend` | #39 |

Three worktrees are also still registered:

```
/Users/leo/code/depiction                                 38d90ff [dev]
/Users/leo/code/depiction-baseline                        ed222b3 (detached)
/Users/leo/code/worktrees/depiction/mild-linden/depiction 38d90ff (detached)
/Users/leo/code/worktrees/depiction/sunny-ledge/depiction 94ee23c [spatial-tools]
```

All three are clean — `git status --short` is empty in each — so **nothing is at risk of being
lost**. That was checked before writing this down.

## Fix sketch

```bash
git worktree remove /Users/leo/code/worktrees/depiction/mild-linden/depiction
git worktree remove /Users/leo/code/worktrees/depiction/sunny-ledge/depiction
git branch -D <the twelve above>
git push origin --delete <the seven remote ones>
```

Keep `/Users/leo/code/depiction-baseline` if you still want the pre-refactor tree for
re-running `docs/refactoring/baseline-diff.md`; it is pinned at `ed222b3`, the last commit
before the migration. If you drop it, note in `baseline-diff.md` how to recreate it, since
that document's reproduction instructions depend on having such a tree.

## Notes

The seven archive tags (`archive/split-packages`, `archive/separate-depiction-io`, …) should
**stay** — they are referenced by `docs/refactoring/ROADMAP.md` as the record of the abandoned
refactorings, and deleting them would break that document.
