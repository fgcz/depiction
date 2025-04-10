import cyclopts

from depiction_tools.cli.cli_imzml import cmd_imzml

app = cyclopts.App()
app.command(cmd_imzml, "imzml")

if __name__ == "__main__":
    app()
