# Two open Dependabot alerts are pinned in `uv.lock`

Severity: **low** | Status: open | Found: 2026-08-10
File: `uv.lock`

## Symptom

`git push` to `dev` reports:

> GitHub found 2 vulnerabilities on fgcz/depiction's default branch (1 moderate, 1 low)

Both come from versions pinned in `uv.lock`:

| package | locked | advisory | fixed in |
|---|---|---|---|
| `setuptools` | 82.0.0 | MANIFEST.in exclusion bypass in sdist via Unicode NFC/NFD collision on macOS APFS/HFS+ | 83.0.0 |
| `pygments` | 2.19.2 | ReDoS via an inefficient regex for GUID matching | 2.20.0 |

Neither is alarming for this project — the setuptools issue needs an attacker-controlled
filename in an sdist build on macOS, and `pygments` is a docs/`rich` dependency that never sees
untrusted input here. They are recorded because of what happens next, not what they are.

## Why it matters for dormancy

1. There is an internal inconsistency worth fixing regardless: `pyproject.toml:115` already
   declares `requires = ["setuptools>=83.0.0"]` for the build system, while the lock resolves
   `setuptools 82.0.0`. The manifest and the lock disagree about the same package.
2. After dormancy, Dependabot keeps filing alerts and pull requests that nobody triages. The
   repository accumulates an unread security backlog, and a future reader cannot tell the
   ignored-because-irrelevant alerts from the ignored-because-abandoned ones.

## Fix sketch

Before closing the lid:

```bash
uv lock --upgrade-package setuptools --upgrade-package pygments
```

then clear the two alerts. That leaves the archive with a clean slate, so any future alert is
unambiguously new.

Then decide explicitly what Dependabot should do afterwards, and record the decision in
`README.md` next to the dormancy note. Either disable it in `.github/dependabot.yml` (honest:
nobody is reading the PRs) or leave it on deliberately so a serious future advisory is at least
visible to whoever inherits this. Silently leaving it on is the option that ages worst.

## Notes

This dimension was missed by the original audit sweep, which checked declared dependencies,
version floors and CI configuration but never queried the advisory surface
(`gh api repos/fgcz/depiction/dependabot/alerts`). Worth re-running that one command before
archiving, since the set will have changed by then.

Related: `021-ci-does-not-install-from-the-lockfile.md` — nothing in CI installs from `uv.lock`
at all, so these pins do not even describe what CI tests.
