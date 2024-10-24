from cyclopts import App

from depiction.tools.calibrate.__main__ import app as calibrate
from depiction.tools.filter_peaks.__main__ import app as filter_peaks
from depiction.tools.cli.cli_generate_ion_images import app as generate_ion_images
from depiction.tools.cli.cli_pick_peaks import app as pick_peaks
from depiction.tools.correct_baseline.__main__ import app as correct_baseline

app = App()
app.command(calibrate, name="calibrate")
app.command(correct_baseline, name="correct-baseline")
app.command(filter_peaks, name="filter-peaks")
app.command(pick_peaks, name="pick-peaks")
app.command(generate_ion_images, name="generate-ion-images")

if __name__ == "__main__":
    app()
