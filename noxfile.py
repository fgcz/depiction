import nox


@nox.session
def tests(session):
    session.install(".[testing]")
    session.run("pytest")
