# snakemake_invoke

A small wrapper for invoking a Snakemake workflow from Python: build the command line from a
pydantic config, run it, and ask for a list of target files.

## Provenance

Vendored on 2026-08-07 from <https://github.com/leoschwarz/snakemake_invoke> at revision
`e7e3c337a022fe4d65894d88b99167fa5bbab67e`, which is the revision `depiction` already
pinned. Same author, same Apache-2.0 licence (see `LICENSE`).

It was vendored because a git dependency on a personal GitHub repository is the one part of
this repository's dependency set that cannot be rebuilt from the repository itself, and the
package is 215 lines used at two call sites. It is not on PyPI, so nothing else can claim
the name. **This copy is the maintained one** — upstream is history, not a source to pull
from.

The first commit of the copy is byte-identical to that revision; formatting to this
repository's conventions (black at 120, ruff) came immediately after, so the two are
separable in the history.

## What is here

- `config.py` — `SnakemakeInvokeConfig`: snakefile path, core count, environment variables,
  and the execution model.
- `invoke/invoke.py` — `SnakemakeInvoke`, the entry point, dispatching on that model.
- `invoke/invoke_subprocess.py` — the path this repository uses. It runs snakemake as a
  subprocess of `sys.executable`, so that the workflow's own `python -m depiction...` shell
  commands resolve against the same interpreter.
- `invoke/invoke_call_function.py` — `snakemake.api` in-process instead. **Unused here**,
  and upstream's own comment says it is not on par with the subprocess path (no report
  generation). It was carried across rather than deleted, so that the copy is the copy.

## Callers

`depiction_targeted_preproc/app_interface/process_chunk.py` — the B-Fabric app entry point,
which turns a chunk directory into a zip of pipeline artifacts. Also
`depiction_cluster_sandbox/run_cluster_sandbox.py`, which is a sandbox rather than a
supported entry point.
