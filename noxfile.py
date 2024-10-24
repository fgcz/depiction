import nox

nox.options.default_venv_backend = "uv"


@nox.session
def lint(session: nox.Session) -> None:
    """Runs the linter."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", "--show-diff-on-failure", *session.posargs)


@nox.session
def tests(session) -> None:
    """Runs the test suite."""
    session.install(".[testing]")
    session.install("pytest-xdist")
    session.run("pytest", "-n", "auto", "--durations=10", "--durations-min=1.0", "tests")


@nox.session
def licensecheck(session) -> None:
    """Runs the license check."""
    session.install("licensecheck")
    session.run("licensecheck")
