# `CreateImzmlPool.write_pool` always raises — a lint autofix ate a local, twice

Severity: **low** | Status: open | Found: 2026-08-10
File: `src/depiction/tools/create_imzml_pool.py:67`

## Symptom

Every call to `write_pool` fails:

```
UndefinedVariableError: local variable 'abs_path' is not defined
```

## Why it happens

```python
with imzml_file.reader() as reader:
    str(imzml_file.imzml_file.absolute())  # <- assignment target stripped
    spectrum_ids = self.pool_source_df.query("abs_path == @abs_path").iloc[0][
        "source_spectrum_id"
    ]
```

Line 67 used to be `abs_path = str(...)`. pandas' `query` resolves `@abs_path` against the
caller's locals; with the assignment gone there is no such local, so the query raises.

This is a lint autofix hazard, not a typo: ruff's F841 ("local variable assigned but never
used") cannot see the reference inside the `query` **string**, so it strips the assignment and
leaves a no-op expression statement. `git log -L 66,70:src/depiction/tools/create_imzml_pool.py`
shows it happening twice:

- `c2aa60f` "more ruff fixes" — stripped `imzml_path =`
- `544d84c` — restored it as `abs_path = str(...)`, fixing the query
- `50a1456` "code style: parallel_ops" — stripped it **again**, which is the current state

## How to reproduce

```python
import pandas as pd

df = pd.DataFrame({"abs_path": ["/a"], "source_spectrum_id": [[1, 2]]})


def f():
    str("/a")
    return df.query("abs_path == @abs_path")


f()  # UndefinedVariableError
```

## Fix sketch

Restore the assignment, and make it autofix-proof by not relying on `query`'s string-scoped
variable lookup:

```python
abs_path = str(imzml_file.imzml_file.absolute())
spectrum_ids = self.pool_source_df.loc[
    self.pool_source_df["abs_path"] == abs_path, "source_spectrum_id"
].iloc[0]
```

The method has no callers anywhere in the repo, so deleting `CreateImzmlPool` entirely is an
equally defensible option and takes less time.

## Notes

Worth a grep for the same pattern elsewhere before dormancy:
`grep -rn 'query(' --include=*.py src/` and check each `@name` still has a live local.
