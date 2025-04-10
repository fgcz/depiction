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
    testfiles = session.posargs if session.posargs else ["tests"]
    session.install(".[testing]")
    session.install("pytest-xdist")
    session.run("pytest", "-n", "auto", "--durations=10", "--durations-min=1.0", *testfiles)


@nox.session
def licensecheck(session) -> None:
    """Runs the license check."""
    session.install("licensecheck")
    session.run("licensecheck")


@nox.session(default=False)
def system_tests(session) -> None:
    """Runs the system test - slow"""
    session.install(".[testing]")
    session.run("pytest", "--durations=10", "system_tests")


@nox.session(default=False)
def tests_structure(session) -> None:
    """Performs a check on the test structure."""
    session.install("check-tests-structure")
    session.run("check-tests-structure", "hook")
