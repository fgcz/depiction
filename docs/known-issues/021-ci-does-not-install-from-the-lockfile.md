# CI never installs from `uv.lock`, and two default sessions depend on the network

Severity: **low** | Status: **accepted — documented** | Found: 2026-08-10
Files: `noxfile.py`, `.github/workflows/pr-checks.yml:25`

## Symptom

Two facts that together mean CI will eventually go red for reasons unrelated to any commit:

1. **The lockfile is inert.** Every nox session installs with `session.install(".[testing]")`
   — a fresh PyPI resolve. Nothing anywhere runs `uv sync`, `--locked` or `--frozen`:

   ```
   $ grep -rn 'uv.lock\|--locked\|uv sync\|--frozen' noxfile.py .github/ .pre-commit-config.yaml README.md
   README.md:37:    uv sync --extra dev
   ```

   So the 357-package lock committed "for reproducibility" reproduces nothing in CI, and there
   is no known-good resolution to fall back to when an upstream release breaks the build.

2. **Two default sessions need the network.** `docs` (sphinx `-W`, intersphinx fetches remote
   inventories) and `licensecheck` (resolves all 357 packages from PyPI) are both default
   sessions, so bare `nox` in CI runs them on every push. A moved `objects.inv` or a changed
   upstream license classifier turns the check red.

## Why it is filed as accepted rather than fixed

Both were deliberate, and both matter mainly to someone *resuming* development — the scenario
dormancy rules out. If nobody opens pull requests, red CI costs nothing. The `docs` session's
network dependency is already documented in its own docstring, and `tests_structure` /
`system_tests` are explicitly `default=False`, so the default set is a considered choice rather
than an oversight.

## What to do instead

Nothing, unless you want the badge to stay green. If you do, the minimal version is one extra
job:

```yaml
- run: uv sync --locked --extra testing && uv run pytest
```

which makes a red "fresh resolve" distinguishable from genuinely broken code, plus
`@nox.session(default=False)` on `docs` and `licensecheck`.

Alternatively, disable the workflow's `push` trigger when you archive the repo, which achieves
the same thing for free.

## Notes

The workflow also caches `.test-data` keyed on `hashFiles('tests/real_data/datasets.py')` and
downloads a 59 MB public acquisition. If that URL rots, `system_tests` fails at the fetch step
rather than in a test — worth knowing before debugging it.
