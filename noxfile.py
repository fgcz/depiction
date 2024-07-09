import nox

nox.options.default_venv_backend = "uv"


@nox.session(reuse_venv=True)
def lint(session: nox.Session) -> None:
    """Runs the linter."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", "--show-diff-on-failure", *session.posargs)


@nox.session
def tests(session) -> None:
    """Runs the test suite."""
    session.install(".[testing]")
    session.run("pytest")
