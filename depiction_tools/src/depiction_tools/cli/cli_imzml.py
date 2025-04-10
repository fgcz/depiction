# TODO i wonder if we should keep tool impl and CLI separate or the cli module is just a thin wrapper around the main
#      functions available in the individual tools
import cyclopts

from depiction_tools.imzml.verify import cmd_imzml_verify
from depiction_tools.imzml.cutout_rectangular_region import cmd_imzml_cutout_rectangular_region

cmd_imzml = cyclopts.App(help="Simple operations on imzML files")
cmd_imzml.command(cmd_imzml_verify, "verify")
cmd_imzml.command(cmd_imzml_cutout_rectangular_region, "cutout-rectangular-region")
