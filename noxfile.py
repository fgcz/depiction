import nox
from pathlib import Path

nox.options.default_venv_backend = "uv"

# Get all package directories with pyproject.toml
# PACKAGES = [str(p.parent) for p in Path(".").glob("*/pyproject.toml")]
PACKAGES = ["depiction_io", "depiction_tools"]


@nox.session
@nox.parametrize("pkg", PACKAGES)
def test(session, pkg):
    """Run tests for a specific package."""
    session.chdir(pkg)
    session.install(".[test]")  # Install package with test extras
    session.run("pytest")


# @nox.session
# def lint_all(session):
#    """Lint all packages at once."""
#    session.install("flake8")
#    session.run("flake8", *PACKAGES)
