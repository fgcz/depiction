# README claims active development and no document says how to run the pipeline

Severity: **medium** | Status: open | Found: 2026-08-10
File: `README.md:5,22`

## Symptom

Two problems for anyone arriving after this repo goes quiet:

1. The front page says *"The full pipeline is also in the process of being developed"* and
   *"This project is in an early state of development. If you are interested, it's best to
   reach out to us"* — inviting contact with a maintainer who will not be there, and implying
   active development that has stopped. The archive decision is recorded only at
   `docs/refactoring/ROADMAP.md:33`, which nobody reads first.
2. Nothing documents how to actually run the pipeline. The real entry point is
   `src/depiction_targeted_preproc/app_interface/process_chunk.py` operating on a chunk
   directory (`params.yml` + `config/` + `raw.imzML`) — not a console script, and not
   mentioned in any markdown file:

```
$ git grep -n 'params.yml\|requested_artifacts' -- '*.md'
(no output)
```

For a dormant repo this is the highest-leverage item on the whole list: everything else is a
bug a reader can find, this is the thing that stops them starting.

## Fix sketch

Two edits to `README.md`:

- Replace the two status sentences with an honest paragraph: the project is dormant/archived
  as of 2026-08, what is supported (the targeted preprocessing pipeline, `depiction_io`) versus
  experimental (clustering, the sandbox, `workflow/exp/`), and where known gaps are recorded
  (this directory, plus `docs/refactoring/ROADMAP.md`).
- Add a short "Running the pipeline" section: the chunk-directory shape, `process_chunk`, the
  `PipelineArtifact` enum as the list of things you can ask for, and a pointer to
  `system_tests/fixtures.py` for a worked example.

## Notes

`system_tests/README.md` does contain a working end-to-end recipe (fetch the `mouse_kidney`
fixture, `nox -s system_tests`), so this is not total darkness — but a stranger has to find it
by accident. Linking it from the front page is most of the fix.

Related open issue: #17 ("Documentation"), which lists installation / configuration / editing
the workflow as the three pieces worth writing.
