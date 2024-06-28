import nox

nox.options.default_venv_backend = "uv"


@nox.session
def tests(session):
    session.install(".[testing]")
    session.run("pytest")
