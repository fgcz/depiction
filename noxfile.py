from pathlib import Path

import nox

nox.options.default_venv_backend = "uv"


@nox.session
def lint(session: nox.Session) -> None:
    """Runs the linter."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", "--show-diff-on-failure", *session.posargs)


@nox.session
def tests_depiction(session) -> None:
    """Runs the test suite of the `depiction` package."""
    testfiles = session.posargs if session.posargs else ["tests"]
    session.install(".[testing]")
    session.install("pytest-xdist")
    session.run("pytest", "-n", "auto", "--durations=10", "--durations-min=1.0", *testfiles)


@nox.session
def tests_depiction_io(session) -> None:
    """Runs the test suite of the `depiction_io` package.

    Installs only `depiction_io`, so that an accidental dependency on `depiction`
    fails here rather than being masked by the parent's environment.
    """
    session.install("./pkgs/depiction_io[testing]")
    session.install("pytest-xdist")
    # posargs are resolved *before* the chdir, so paths can be given relative to the
    # repository root as in every other session; otherwise `nox -s tests_depiction_io --
    # pkgs/depiction_io/tests/unit/imzml` would resolve against the wrong base.
    testfiles = [str(Path(arg).resolve()) for arg in session.posargs] or ["tests"]
    session.chdir("pkgs/depiction_io")
    session.run("pytest", "-n", "auto", "--durations=10", "--durations-min=1.0", *testfiles)


@nox.session
def docs(session) -> None:
    """Builds the Sphinx documentation, treating warnings as errors.

    Nothing built these docs before, which is how an autodoc reference to a module that had
    been deleted months earlier survived unnoticed. `-W` is what makes that impossible to
    repeat. Note that intersphinx fetches remote inventories, so this session needs network
    access and will fail if one of the referenced sites is unreachable.
    """
    session.install(".[doc]")
    session.run("sphinx-build", "-W", "-b", "html", "docs", "docs/_build/html", *session.posargs)


@nox.session
def licensecheck(session) -> None:
    """Runs the license check."""
    session.install("licensecheck")
    # depiction_io is skipped because it is our own workspace member (same license as the
    # root) and because licensecheck's resolver cannot parse the `-e file:///...` entry
    # that uv emits for a workspace dependency.
    session.run("licensecheck", "--skip-dependencies", "llvmlite", "depiction_io")


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
